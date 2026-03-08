const path = require('path');
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const PORT = process.env.PORT || 3000;
const ARENA = { width: 1400, height: 900 };
const PLAYER_SPEED = 260;
const DASH_SPEED = 640;
const DASH_DURATION_MS = 140;
const DASH_COOLDOWN_MS = 1800;
const ATTACK_RANGE = 84;
const ATTACK_DAMAGE = 24;
const ATTACK_COOLDOWN_MS = 400;
const SPECIAL_RANGE = 180;
const SPECIAL_DAMAGE = 35;
const SPECIAL_COOLDOWN_MS = 5000;
const HAZARD_DAMAGE_PER_SECOND = 26;
const MAX_HEALTH = 100;
const ROUND_MS = 90_000;
const RESPAWN_MS = 3000;

const rooms = new Map();

function randomInRange(min, max) {
  return min + Math.random() * (max - min);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function randomSpawn() {
  return {
    x: randomInRange(90, ARENA.width - 90),
    y: randomInRange(90, ARENA.height - 90)
  };
}

function code() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let value = '';
  for (let i = 0; i < 4; i += 1) {
    value += chars[Math.floor(Math.random() * chars.length)];
  }
  return value;
}

function createRoom(hostSocketId, hostName) {
  let roomCode = code();
  while (rooms.has(roomCode)) roomCode = code();

  const spawn = randomSpawn();
  const room = {
    code: roomCode,
    hostSocketId,
    phase: 'lobby',
    startedAt: null,
    endsAt: null,
    players: {},
    hazard: { x: ARENA.width / 2, y: ARENA.height / 2, radius: 90 },
    powerUp: { x: ARENA.width / 2 + 240, y: ARENA.height / 2, active: true, radius: 20, nextSpawnAt: 0 },
    killFeed: []
  };

  room.players[hostSocketId] = {
    id: hostSocketId,
    name: hostName || 'Host',
    x: spawn.x,
    y: spawn.y,
    vx: 0,
    vy: 0,
    hp: MAX_HEALTH,
    alive: true,
    respawnAt: 0,
    score: 0,
    input: { up: false, down: false, left: false, right: false, dash: false, attack: false, special: false, angle: 0 },
    lastDashAt: -99999,
    dashingUntil: 0,
    lastAttackAt: -99999,
    lastSpecialAt: -99999,
    boostUntil: 0
  };

  rooms.set(roomCode, room);
  return room;
}

function roomSummary(room) {
  return {
    code: room.code,
    phase: room.phase,
    hostSocketId: room.hostSocketId,
    arena: ARENA,
    startedAt: room.startedAt,
    endsAt: room.endsAt,
    hazard: room.hazard,
    powerUp: room.powerUp,
    players: Object.values(room.players).map((p) => ({
      id: p.id,
      name: p.name,
      x: p.x,
      y: p.y,
      hp: p.hp,
      alive: p.alive,
      score: p.score,
      boostUntil: p.boostUntil,
      lastDashAt: p.lastDashAt,
      lastSpecialAt: p.lastSpecialAt
    }))
  };
}

// Networking: this is the server-authoritative simulation entry point.
io.on('connection', (socket) => {
  socket.on('createRoom', ({ name }, reply = () => {}) => {
    const room = createRoom(socket.id, name);
    socket.join(room.code);
    socket.data.roomCode = room.code;
    reply({ ok: true, room: roomSummary(room), playerId: socket.id });
    io.to(room.code).emit('roomUpdated', roomSummary(room));
  });

  socket.on('joinRoom', ({ roomCode, name }, reply = () => {}) => {
    const room = rooms.get((roomCode || '').toUpperCase());
    if (!room) return reply({ ok: false, error: 'Room not found' });
    if (room.phase !== 'lobby') return reply({ ok: false, error: 'Match already in progress' });
    if (Object.keys(room.players).length >= 4) return reply({ ok: false, error: 'Room full' });

    const spawn = randomSpawn();
    room.players[socket.id] = {
      id: socket.id,
      name: name || `Player ${Object.keys(room.players).length + 1}`,
      x: spawn.x,
      y: spawn.y,
      vx: 0,
      vy: 0,
      hp: MAX_HEALTH,
      alive: true,
      respawnAt: 0,
      score: 0,
      input: { up: false, down: false, left: false, right: false, dash: false, attack: false, special: false, angle: 0 },
      lastDashAt: -99999,
      dashingUntil: 0,
      lastAttackAt: -99999,
      lastSpecialAt: -99999,
      boostUntil: 0
    };

    socket.join(room.code);
    socket.data.roomCode = room.code;
    reply({ ok: true, room: roomSummary(room), playerId: socket.id });
    io.to(room.code).emit('roomUpdated', roomSummary(room));
  });

  socket.on('startMatch', (_, reply = () => {}) => {
    const room = rooms.get(socket.data.roomCode);
    if (!room) return reply({ ok: false, error: 'No room' });
    if (room.hostSocketId !== socket.id) return reply({ ok: false, error: 'Only host can start' });
    if (Object.keys(room.players).length < 2) return reply({ ok: false, error: 'Need at least 2 players' });

    room.phase = 'playing';
    room.startedAt = Date.now();
    room.endsAt = room.startedAt + ROUND_MS;
    room.killFeed = [];
    Object.values(room.players).forEach((p) => {
      const spawn = randomSpawn();
      p.x = spawn.x;
      p.y = spawn.y;
      p.hp = MAX_HEALTH;
      p.alive = true;
      p.respawnAt = 0;
      p.score = 0;
      p.lastDashAt = -99999;
      p.lastAttackAt = -99999;
      p.lastSpecialAt = -99999;
      p.boostUntil = 0;
    });
    io.to(room.code).emit('matchStarted', roomSummary(room));
    reply({ ok: true });
  });

  // Networking: the client only sends inputs; all combat and score outcomes are computed here.
  socket.on('playerInput', (input) => {
    const room = rooms.get(socket.data.roomCode);
    if (!room || !room.players[socket.id]) return;
    room.players[socket.id].input = {
      up: !!input.up,
      down: !!input.down,
      left: !!input.left,
      right: !!input.right,
      dash: !!input.dash,
      attack: !!input.attack,
      special: !!input.special,
      angle: Number.isFinite(input.angle) ? input.angle : 0
    };
  });

  socket.on('disconnect', () => {
    const room = rooms.get(socket.data.roomCode);
    if (!room) return;

    delete room.players[socket.id];
    if (room.hostSocketId === socket.id) {
      room.hostSocketId = Object.keys(room.players)[0] || null;
    }

    if (Object.keys(room.players).length === 0) {
      rooms.delete(room.code);
      return;
    }

    if (room.phase === 'playing' && Object.keys(room.players).length < 2) {
      room.phase = 'lobby';
      room.endsAt = null;
      io.to(room.code).emit('roomUpdated', roomSummary(room));
    } else {
      io.to(room.code).emit('roomUpdated', roomSummary(room));
    }
  });
});

// Combat and game-state simulation loop.
setInterval(() => {
  const now = Date.now();

  rooms.forEach((room) => {
    if (room.phase !== 'playing') return;

    Object.values(room.players).forEach((player) => {
      if (!player.alive) {
        if (now >= player.respawnAt) {
          const spawn = randomSpawn();
          player.alive = true;
          player.hp = MAX_HEALTH;
          player.x = spawn.x;
          player.y = spawn.y;
        }
        return;
      }

      const input = player.input;
      let dx = 0;
      let dy = 0;
      if (input.up) dy -= 1;
      if (input.down) dy += 1;
      if (input.left) dx -= 1;
      if (input.right) dx += 1;

      const magnitude = Math.hypot(dx, dy) || 1;
      const baseSpeed = PLAYER_SPEED * (now < player.boostUntil ? 1.15 : 1);
      let speed = baseSpeed;

      if (input.dash && now - player.lastDashAt > DASH_COOLDOWN_MS) {
        player.lastDashAt = now;
        player.dashingUntil = now + DASH_DURATION_MS;
      }
      if (now < player.dashingUntil) {
        speed = DASH_SPEED;
      }

      player.vx = (dx / magnitude) * speed;
      player.vy = (dy / magnitude) * speed;
      player.x = clamp(player.x + player.vx * (1 / 20), 20, ARENA.width - 20);
      player.y = clamp(player.y + player.vy * (1 / 20), 20, ARENA.height - 20);

      const inHazard = Math.hypot(player.x - room.hazard.x, player.y - room.hazard.y) < room.hazard.radius;
      if (inHazard) {
        player.hp -= HAZARD_DAMAGE_PER_SECOND * (1 / 20);
      }

      if (room.powerUp.active && Math.hypot(player.x - room.powerUp.x, player.y - room.powerUp.y) < room.powerUp.radius + 14) {
        room.powerUp.active = false;
        room.powerUp.nextSpawnAt = now + 12_000;
        player.boostUntil = now + 8_000;
      }

      if (!room.powerUp.active && now >= room.powerUp.nextSpawnAt) {
        room.powerUp.active = true;
        room.powerUp.x = randomInRange(120, ARENA.width - 120);
        room.powerUp.y = randomInRange(120, ARENA.height - 120);
      }

      if (input.attack && now - player.lastAttackAt > ATTACK_COOLDOWN_MS) {
        player.lastAttackAt = now;
        Object.values(room.players).forEach((target) => {
          if (target.id === player.id || !target.alive) return;
          const d = Math.hypot(player.x - target.x, player.y - target.y);
          if (d <= ATTACK_RANGE) {
            target.hp -= ATTACK_DAMAGE;
            if (target.hp <= 0) {
              target.alive = false;
              target.respawnAt = now + RESPAWN_MS;
              player.score += 1;
            }
          }
        });
      }

      if (input.special && now - player.lastSpecialAt > SPECIAL_COOLDOWN_MS) {
        player.lastSpecialAt = now;
        const bonus = now < player.boostUntil ? 12 : 0;
        Object.values(room.players).forEach((target) => {
          if (target.id === player.id || !target.alive) return;
          const toTargetX = target.x - player.x;
          const toTargetY = target.y - player.y;
          const dist = Math.hypot(toTargetX, toTargetY);
          if (dist > SPECIAL_RANGE) return;
          const targetAngle = Math.atan2(toTargetY, toTargetX);
          const angleDelta = Math.abs(Math.atan2(Math.sin(targetAngle - input.angle), Math.cos(targetAngle - input.angle)));
          if (angleDelta <= Math.PI / 3) {
            target.hp -= SPECIAL_DAMAGE + bonus;
            if (target.hp <= 0) {
              target.alive = false;
              target.respawnAt = now + RESPAWN_MS;
              player.score += 1;
            }
          }
        });
      }

      if (player.hp <= 0 && player.alive) {
        player.alive = false;
        player.respawnAt = now + RESPAWN_MS;
      }
    });

    if (now >= room.endsAt) {
      room.phase = 'results';
      const ranking = Object.values(room.players)
        .map((p) => ({ id: p.id, name: p.name, score: p.score }))
        .sort((a, b) => b.score - a.score);
      io.to(room.code).emit('matchEnded', { ranking, roomCode: room.code });
      room.phase = 'lobby';
      room.endsAt = null;
    }

    io.to(room.code).emit('state', roomSummary(room));
  });
}, 50);

app.use(express.static(path.resolve(__dirname, '../../client')));

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Arena brawler server running on http://localhost:${PORT}`);
});
