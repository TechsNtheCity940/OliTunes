const path = require('path');
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const PORT = process.env.PORT || 3000;
const ARENA = { width: 1400, height: 900 };
const PLAYER_SPEED = 240;
const DASH_SPEED = 620;
const DASH_DURATION_MS = 150;
const DASH_COOLDOWN_MS = 1800;
const ATTACK_RANGE = 82;
const ATTACK_DAMAGE = 22;
const ATTACK_COOLDOWN_MS = 400;
const ELEMENT_COOLDOWN_MS = 5000;
const ELEMENT_BASE_DAMAGE = 28;
const ELEMENT_RADIUS = 105;
const ELEMENT_LIFE_MS = 1200;
const HAZARD_DAMAGE_PER_SECOND = 24;
const MAX_HEALTH = 100;
const ROUND_MS = 90_000;
const RESPAWN_MS = 3000;
const KILL_HEAL = 14;
const MAX_RANK = 99;
const BOT_COUNT = 4;

const CLASS_DEFS = {
  Earth: { color: 0x9d7a53, accent: 0x5d472f, elementName: 'Earthquake', elementColor: 0xb58a63 },
  Fire: { color: 0xe8613c, accent: 0x9b2a1b, elementName: 'Meteor', elementColor: 0xff813f },
  Wind: { color: 0x70d7d4, accent: 0x2e7f8d, elementName: 'Tornado', elementColor: 0x8bf3ff },
  Water: { color: 0x4b7dff, accent: 0x1f4eb7, elementName: 'Ice Spike', elementColor: 0x7db5ff }
};

const COSMETIC_VARIANTS = ['stripe', 'spikes', 'halo'];
const BOT_DIFFICULTY = {
  easy: { aimError: 0.6, attackDistance: 62, elementDistance: 130, reactionChance: 0.45 },
  normal: { aimError: 0.3, attackDistance: 72, elementDistance: 155, reactionChance: 0.7 },
  hard: { aimError: 0.12, attackDistance: 80, elementDistance: 175, reactionChance: 0.9 }
};

const rooms = new Map();

function randomInRange(min, max) {
  return min + Math.random() * (max - min);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function randomSpawn() {
  return { x: randomInRange(90, ARENA.width - 90), y: randomInRange(90, ARENA.height - 90) };
}

function code() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let value = '';
  for (let i = 0; i < 4; i += 1) value += chars[Math.floor(Math.random() * chars.length)];
  return value;
}

function safeClass(className) {
  return CLASS_DEFS[className] ? className : 'Earth';
}

function safeAppearance(className, appearance = {}) {
  const selected = COSMETIC_VARIANTS.includes(appearance.cosmetic) ? appearance.cosmetic : COSMETIC_VARIANTS[0];
  return {
    cosmetic: selected,
    tint: Number.isFinite(appearance.tint) ? appearance.tint : CLASS_DEFS[className].color,
    winAuraLevel: clamp(appearance.winAuraLevel || 0, 0, 20)
  };
}

function rankFromKills(totalKills) {
  return clamp(1 + totalKills, 1, MAX_RANK);
}

function newPlayer({ id, name, className, appearance, isBot = false, botDifficulty = 'normal' }) {
  const spawn = randomSpawn();
  const classSafe = safeClass(className);
  const botSafe = BOT_DIFFICULTY[botDifficulty] ? botDifficulty : 'normal';

  return {
    id,
    name: name || (isBot ? 'Bot' : 'Player'),
    className: classSafe,
    classMeta: CLASS_DEFS[classSafe],
    appearance: safeAppearance(classSafe, appearance),
    isBot,
    botDifficulty: botSafe,
    x: spawn.x,
    y: spawn.y,
    vx: 0,
    vy: 0,
    hp: MAX_HEALTH,
    alive: true,
    respawnAt: 0,
    score: 0,
    totalKills: 0,
    rank: 1,
    wins: 0,
    input: { up: false, down: false, left: false, right: false, dash: false, attack: false, special: false, angle: 0, specialTargetX: spawn.x, specialTargetY: spawn.y },
    lastDashAt: -99999,
    dashingUntil: 0,
    lastAttackAt: -99999,
    lastSpecialAt: -99999,
    boostUntil: 0
  };
}

function createRoom(hostSocketId, hostConfig = {}) {
  let roomCode = code();
  while (rooms.has(roomCode)) roomCode = code();

  const room = {
    code: roomCode,
    hostSocketId,
    phase: 'lobby',
    startedAt: null,
    endsAt: null,
    players: {},
    botDifficulty: BOT_DIFFICULTY[hostConfig.botDifficulty] ? hostConfig.botDifficulty : 'normal',
    hazard: { x: ARENA.width / 2, y: ARENA.height / 2, radius: 95 },
    powerUp: { x: ARENA.width / 2 + 240, y: ARENA.height / 2, active: true, radius: 20, nextSpawnAt: 0 },
    elementZones: []
  };

  room.players[hostSocketId] = newPlayer({
    id: hostSocketId,
    name: hostConfig.name,
    className: hostConfig.className,
    appearance: hostConfig.appearance
  });

  rooms.set(roomCode, room);
  return room;
}

function roomSummary(room) {
  return {
    code: room.code,
    phase: room.phase,
    hostSocketId: room.hostSocketId,
    arena: ARENA,
    botDifficulty: room.botDifficulty,
    startedAt: room.startedAt,
    endsAt: room.endsAt,
    hazard: room.hazard,
    powerUp: room.powerUp,
    elementZones: room.elementZones,
    players: Object.values(room.players).map((p) => ({
      id: p.id,
      name: p.name,
      className: p.className,
      classMeta: p.classMeta,
      appearance: p.appearance,
      isBot: p.isBot,
      botDifficulty: p.botDifficulty,
      x: p.x,
      y: p.y,
      hp: p.hp,
      alive: p.alive,
      score: p.score,
      totalKills: p.totalKills,
      rank: p.rank,
      wins: p.wins,
      boostUntil: p.boostUntil,
      lastDashAt: p.lastDashAt,
      lastSpecialAt: p.lastSpecialAt
    }))
  };
}

function addBotsToRoom(room) {
  const existing = Object.keys(room.players).length;
  for (let i = existing; i < BOT_COUNT; i += 1) {
    const className = Object.keys(CLASS_DEFS)[i % 4];
    const id = `bot_${room.code}_${i}`;
    room.players[id] = newPlayer({
      id,
      isBot: true,
      botDifficulty: room.botDifficulty,
      className,
      name: `Bot ${i + 1}`,
      appearance: { cosmetic: COSMETIC_VARIANTS[i % COSMETIC_VARIANTS.length] }
    });
  }
}

function removeBotsFromLobby(room) {
  Object.keys(room.players).forEach((id) => {
    if (room.players[id].isBot && room.phase === 'lobby') {
      delete room.players[id];
    }
  });
}

function registerKill(attacker, target) {
  target.alive = false;
  target.respawnAt = Date.now() + RESPAWN_MS;
  attacker.score += 1;
  attacker.totalKills += 1;
  attacker.rank = rankFromKills(attacker.totalKills);
  attacker.hp = clamp(attacker.hp + KILL_HEAL, 0, MAX_HEALTH);
}

function botThink(room, bot, now) {
  const botAI = BOT_DIFFICULTY[bot.botDifficulty] || BOT_DIFFICULTY.normal;
  const enemies = Object.values(room.players).filter((p) => p.id !== bot.id && p.alive);
  if (!enemies.length || Math.random() > botAI.reactionChance) {
    bot.input = { up: false, down: false, left: false, right: false, dash: false, attack: false, special: false, angle: 0, specialTargetX: bot.x, specialTargetY: bot.y };
    return;
  }

  enemies.sort((a, b) => Math.hypot(a.x - bot.x, a.y - bot.y) - Math.hypot(b.x - bot.x, b.y - bot.y));
  const target = enemies[0];
  const dx = target.x - bot.x;
  const dy = target.y - bot.y;
  const dist = Math.hypot(dx, dy) || 1;

  const input = {
    up: dy < -10,
    down: dy > 10,
    left: dx < -10,
    right: dx > 10,
    dash: dist > 220 && now - bot.lastDashAt > DASH_COOLDOWN_MS,
    attack: dist <= botAI.attackDistance && now - bot.lastAttackAt > ATTACK_COOLDOWN_MS,
    special: dist <= botAI.elementDistance && now - bot.lastSpecialAt > ELEMENT_COOLDOWN_MS,
    angle: Math.atan2(dy, dx) + randomInRange(-botAI.aimError, botAI.aimError),
    specialTargetX: target.x + randomInRange(-20, 20),
    specialTargetY: target.y + randomInRange(-20, 20)
  };

  bot.input = input;
}

// Networking: this server owns room, combat, score, rank and timeline authority.
io.on('connection', (socket) => {
  socket.on('createRoom', (payload, reply = () => {}) => {
    const room = createRoom(socket.id, payload || {});
    socket.join(room.code);
    socket.data.roomCode = room.code;
    reply({ ok: true, room: roomSummary(room), playerId: socket.id });
    io.to(room.code).emit('roomUpdated', roomSummary(room));
  });

  socket.on('joinRoom', (payload, reply = () => {}) => {
    const room = rooms.get((payload?.roomCode || '').toUpperCase());
    if (!room) return reply({ ok: false, error: 'Room not found' });
    if (room.phase !== 'lobby') return reply({ ok: false, error: 'Match already in progress' });

    removeBotsFromLobby(room);
    const humanCount = Object.values(room.players).filter((p) => !p.isBot).length;
    if (humanCount >= 4) return reply({ ok: false, error: 'Room full' });

    room.players[socket.id] = newPlayer({
      id: socket.id,
      name: payload?.name,
      className: payload?.className,
      appearance: payload?.appearance
    });

    socket.join(room.code);
    socket.data.roomCode = room.code;
    reply({ ok: true, room: roomSummary(room), playerId: socket.id });
    io.to(room.code).emit('roomUpdated', roomSummary(room));
  });

  socket.on('startMatch', (_, reply = () => {}) => {
    const room = rooms.get(socket.data.roomCode);
    if (!room) return reply({ ok: false, error: 'No room' });
    if (room.hostSocketId !== socket.id) return reply({ ok: false, error: 'Only host can start' });

    const humanCount = Object.values(room.players).filter((p) => !p.isBot).length;
    if (humanCount < 1) return reply({ ok: false, error: 'Need at least 1 player' });

    addBotsToRoom(room);
    room.phase = 'playing';
    room.startedAt = Date.now();
    room.endsAt = room.startedAt + ROUND_MS;
    room.elementZones = [];

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

  // Networking: clients send intent only; server executes movement/combat/element effects.
  socket.on('playerInput', (input = {}) => {
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
      angle: Number.isFinite(input.angle) ? input.angle : 0,
      specialTargetX: Number.isFinite(input.specialTargetX) ? input.specialTargetX : room.players[socket.id].x,
      specialTargetY: Number.isFinite(input.specialTargetY) ? input.specialTargetY : room.players[socket.id].y
    };
  });

  socket.on('setBotDifficulty', ({ difficulty }, reply = () => {}) => {
    const room = rooms.get(socket.data.roomCode);
    if (!room) return reply({ ok: false, error: 'No room' });
    if (room.hostSocketId !== socket.id) return reply({ ok: false, error: 'Only host can set bot difficulty' });
    if (!BOT_DIFFICULTY[difficulty]) return reply({ ok: false, error: 'Invalid difficulty' });

    room.botDifficulty = difficulty;
    Object.values(room.players).forEach((p) => {
      if (p.isBot) p.botDifficulty = difficulty;
    });

    io.to(room.code).emit('roomUpdated', roomSummary(room));
    reply({ ok: true });
  });

  socket.on('disconnect', () => {
    const room = rooms.get(socket.data.roomCode);
    if (!room) return;

    delete room.players[socket.id];
    if (room.hostSocketId === socket.id) {
      const humans = Object.values(room.players).filter((p) => !p.isBot);
      room.hostSocketId = humans[0]?.id || null;
    }

    const humansLeft = Object.values(room.players).filter((p) => !p.isBot);
    if (humansLeft.length === 0) {
      rooms.delete(room.code);
      return;
    }

    if (room.phase === 'playing' && humansLeft.length === 1) {
      addBotsToRoom(room);
    }

    io.to(room.code).emit('roomUpdated', roomSummary(room));
  });
});

// Combat + game-state loop.
setInterval(() => {
  const now = Date.now();

  rooms.forEach((room) => {
    if (room.phase !== 'playing') return;

    Object.values(room.players).forEach((player) => {
      if (player.isBot) botThink(room, player, now);
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
      if (now < player.dashingUntil) speed = DASH_SPEED;

      player.vx = (dx / magnitude) * speed;
      player.vy = (dy / magnitude) * speed;
      player.x = clamp(player.x + player.vx * (1 / 20), 20, ARENA.width - 20);
      player.y = clamp(player.y + player.vy * (1 / 20), 20, ARENA.height - 20);

      const inHazard = Math.hypot(player.x - room.hazard.x, player.y - room.hazard.y) < room.hazard.radius;
      if (inHazard) player.hp -= HAZARD_DAMAGE_PER_SECOND * (1 / 20);

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
          if (Math.hypot(player.x - target.x, player.y - target.y) <= ATTACK_RANGE) {
            target.hp -= ATTACK_DAMAGE;
            if (target.hp <= 0) registerKill(player, target);
          }
        });
      }

      // Combat: class-based elemental AoE attacks are equal-strength and only change visuals/theme.
      if (input.special && now - player.lastSpecialAt > ELEMENT_COOLDOWN_MS) {
        player.lastSpecialAt = now;
        room.elementZones.push({
          ownerId: player.id,
          className: player.className,
          x: clamp(input.specialTargetX, 30, ARENA.width - 30),
          y: clamp(input.specialTargetY, 30, ARENA.height - 30),
          radius: ELEMENT_RADIUS,
          damage: ELEMENT_BASE_DAMAGE + (now < player.boostUntil ? 8 : 0),
          expiresAt: now + ELEMENT_LIFE_MS,
          color: player.classMeta.elementColor,
          label: player.classMeta.elementName
        });
      }

      if (player.hp <= 0 && player.alive) {
        player.alive = false;
        player.respawnAt = now + RESPAWN_MS;
      }
    });

    room.elementZones = room.elementZones.filter((zone) => zone.expiresAt > now);
    room.elementZones.forEach((zone) => {
      const owner = room.players[zone.ownerId];
      if (!owner) return;
      Object.values(room.players).forEach((target) => {
        if (!target.alive || target.id === zone.ownerId) return;
        if (Math.hypot(target.x - zone.x, target.y - zone.y) <= zone.radius) {
          target.hp -= zone.damage * (1 / 20);
          if (target.hp <= 0 && target.alive) registerKill(owner, target);
        }
      });
    });

    if (now >= room.endsAt) {
      const ranking = Object.values(room.players)
        .map((p) => ({ id: p.id, name: p.name, score: p.score, className: p.className, rank: p.rank, wins: p.wins, isBot: p.isBot }))
        .sort((a, b) => b.score - a.score);

      const winner = ranking[0] ? room.players[ranking[0].id] : null;
      if (winner) {
        winner.wins += 1;
        winner.appearance.winAuraLevel = clamp(winner.appearance.winAuraLevel + 1, 0, 20);
      }

      io.to(room.code).emit('matchEnded', { ranking, roomCode: room.code, winnerId: winner?.id || null });
      room.phase = 'lobby';
      room.endsAt = null;
      room.elementZones = [];
      removeBotsFromLobby(room);
      io.to(room.code).emit('roomUpdated', roomSummary(room));
      return;
    }

    io.to(room.code).emit('state', roomSummary(room));
  });
}, 50);

app.use(express.static(path.resolve(__dirname, '../../client')));
server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Arena brawler server running on http://localhost:${PORT}`);
});
