const socket = io();

const CLASS_OPTIONS = ['Earth', 'Fire', 'Wind', 'Water'];
const COSMETIC_OPTIONS = ['stripe', 'spikes', 'halo'];

const shared = {
  playerId: null,
  room: null,
  latestState: null,
  profile: null,
  input: {
    up: false,
    down: false,
    left: false,
    right: false,
    dash: false,
    attack: false,
    special: false,
    angle: 0,
    specialTargetX: 0,
    specialTargetY: 0
  }
};

function promptProfile() {
  const name = (window.prompt('Character name?', 'Player') || 'Player').slice(0, 18);
  const classNameRaw = (window.prompt('Class: Earth / Fire / Wind / Water', 'Earth') || 'Earth').trim();
  const className = CLASS_OPTIONS.includes(classNameRaw) ? classNameRaw : 'Earth';
  const cosmeticRaw = (window.prompt('Cosmetic: stripe / spikes / halo', 'stripe') || 'stripe').trim();
  const cosmetic = COSMETIC_OPTIONS.includes(cosmeticRaw) ? cosmeticRaw : 'stripe';
  return { name, className, appearance: { cosmetic } };
}

class TitleScene extends Phaser.Scene {
  constructor() {
    super('TitleScene');
  }

  create() {
    this.add.text(700, 220, 'ARENA BRAWLER PROTOTYPE', { fontSize: '44px', color: '#ffffff' }).setOrigin(0.5);
    this.add.text(700, 300, 'Create a class-based character, fight players + bots, and rank up to 99.', { fontSize: '22px', color: '#d4dcff' }).setOrigin(0.5);
    this.add.text(700, 350, 'WASD Move | Space Dash | LMB Melee | RMB Elemental AoE', { fontSize: '22px', color: '#d4dcff' }).setOrigin(0.5);
    this.add.text(700, 450, 'Press ENTER to continue', { fontSize: '30px', color: '#ffd65c' }).setOrigin(0.5);
    this.input.keyboard.once('keydown-ENTER', () => this.scene.start('MenuScene'));
  }
}

class MenuScene extends Phaser.Scene {
  constructor() {
    super('MenuScene');
  }

  create() {
    this.add.text(700, 140, 'ROOM MENU', { fontSize: '44px', color: '#fff' }).setOrigin(0.5);
    this.add.text(700, 220, 'Press C to Create Room', { fontSize: '30px', color: '#77ff77' }).setOrigin(0.5);
    this.add.text(700, 280, 'Press J to Join Room', { fontSize: '30px', color: '#77ccff' }).setOrigin(0.5);
    this.add.text(700, 340, 'Host can choose bot difficulty: easy / normal / hard', { fontSize: '24px', color: '#f7f7f7' }).setOrigin(0.5);
    this.add.text(700, 390, 'If solo, match auto-fills empty slots with bots.', { fontSize: '24px', color: '#f7f7f7' }).setOrigin(0.5);

    this.input.keyboard.on('keydown-C', () => {
      const profile = promptProfile();
      shared.profile = profile;
      const diffRaw = (window.prompt('Bot difficulty? easy / normal / hard', 'normal') || 'normal').trim().toLowerCase();
      const botDifficulty = ['easy', 'normal', 'hard'].includes(diffRaw) ? diffRaw : 'normal';
      socket.emit('createRoom', { ...profile, botDifficulty }, (response) => {
        if (!response.ok) return;
        shared.playerId = response.playerId;
        shared.room = response.room;
        this.scene.start('LobbyScene');
      });
    });

    this.input.keyboard.on('keydown-J', () => {
      const roomCode = (window.prompt('Room code?') || '').trim().toUpperCase();
      if (!roomCode) return;
      const profile = promptProfile();
      shared.profile = profile;
      socket.emit('joinRoom', { roomCode, ...profile }, (response) => {
        if (!response.ok) return window.alert(response.error);
        shared.playerId = response.playerId;
        shared.room = response.room;
        this.scene.start('LobbyScene');
      });
    });
  }
}

class LobbyScene extends Phaser.Scene {
  constructor() {
    super('LobbyScene');
  }

  create() {
    this.add.text(700, 100, 'Lobby', { fontSize: '42px', color: '#ffffff' }).setOrigin(0.5);
    this.roomCodeText = this.add.text(700, 160, '', { fontSize: '28px', color: '#ffd65c' }).setOrigin(0.5);
    this.diffText = this.add.text(700, 210, '', { fontSize: '24px', color: '#9fc2ff' }).setOrigin(0.5);
    this.playerListText = this.add.text(700, 380, '', { fontSize: '24px', color: '#efefef', align: 'center' }).setOrigin(0.5);
    this.help = this.add.text(700, 690, 'S = Start match | D = set bot difficulty (host)', { fontSize: '22px', color: '#9fc2ff' }).setOrigin(0.5);

    this.input.keyboard.on('keydown-S', () => {
      socket.emit('startMatch', {}, (response) => {
        if (!response.ok) window.alert(response.error);
      });
    });

    this.input.keyboard.on('keydown-D', () => {
      const diffRaw = (window.prompt('Set bot difficulty: easy / normal / hard', shared.room?.botDifficulty || 'normal') || 'normal').trim().toLowerCase();
      const difficulty = ['easy', 'normal', 'hard'].includes(diffRaw) ? diffRaw : 'normal';
      socket.emit('setBotDifficulty', { difficulty }, (response) => {
        if (!response.ok) window.alert(response.error);
      });
    });

    socket.off('roomUpdated');
    socket.on('roomUpdated', (roomState) => {
      shared.room = roomState;
      this.renderRoom(roomState);
    });

    socket.off('matchStarted');
    socket.on('matchStarted', (roomState) => {
      shared.latestState = roomState;
      this.scene.start('ArenaScene');
    });

    this.renderRoom(shared.room);
  }

  renderRoom(roomState) {
    if (!roomState) return;
    this.roomCodeText.setText(`Room Code: ${roomState.code}`);
    this.diffText.setText(`Bot Difficulty: ${roomState.botDifficulty}`);
    const lines = roomState.players
      .map((p) => `${p.id === roomState.hostSocketId ? '[HOST] ' : ''}${p.name} (${p.className}${p.isBot ? ' BOT' : ''})`)
      .join('\n');
    this.playerListText.setText(lines || 'Waiting...');
  }
}

class ArenaScene extends Phaser.Scene {
  constructor() {
    super('ArenaScene');
  }

  create() {
    this.playerGraphics = new Map();
    this.elementGraphics = [];
    this.add.rectangle(700, 450, 1400, 900, 0x1e2433, 1);
    this.hazardCircle = this.add.circle(0, 0, 90, 0xff4956, 0.35);
    this.powerUpGraphic = this.add.rectangle(0, 0, 26, 26, 0x60ff95, 0.95);
    this.timerText = this.add.text(20, 14, 'Time: 90', { fontSize: '28px', color: '#fff' });
    this.scoreboardText = this.add.text(20, 48, '', { fontSize: '18px', color: '#fff' });
    this.myStats = this.add.text(20, 280, '', { fontSize: '18px', color: '#ffd65c' });

    this.keys = this.input.keyboard.addKeys('W,A,S,D,SPACE');
    this.input.on('pointerdown', (pointer) => {
      if (pointer.rightButtonDown()) {
        shared.input.special = true;
        shared.input.specialTargetX = pointer.worldX;
        shared.input.specialTargetY = pointer.worldY;
      } else {
        shared.input.attack = true;
      }
    });
    this.input.mouse?.disableContextMenu();

    socket.off('state');
    socket.on('state', (state) => {
      shared.latestState = state;
      if (state.phase === 'lobby') this.scene.start('LobbyScene');
    });

    socket.off('matchEnded');
    socket.on('matchEnded', ({ ranking, roomCode, winnerId }) => {
      this.scene.start('ResultsScene', { ranking, roomCode, winnerId });
    });
  }

  update() {
    const state = shared.latestState;
    if (!state) return;

    // Networking: send intent only, server is authoritative for combat + score + ranks.
    const me = state.players.find((p) => p.id === shared.playerId);
    if (me) {
      shared.input.up = this.keys.W.isDown;
      shared.input.down = this.keys.S.isDown;
      shared.input.left = this.keys.A.isDown;
      shared.input.right = this.keys.D.isDown;
      shared.input.dash = Phaser.Input.Keyboard.JustDown(this.keys.SPACE);
      const pointer = this.input.activePointer;
      shared.input.angle = Phaser.Math.Angle.Between(me.x, me.y, pointer.worldX, pointer.worldY);

      socket.emit('playerInput', shared.input);
      shared.input.attack = false;
      shared.input.special = false;
    }

    this.renderState(state, me);
  }

  renderState(state, me) {
    this.hazardCircle.setPosition(state.hazard.x, state.hazard.y);
    this.powerUpGraphic.setPosition(state.powerUp.x, state.powerUp.y);
    this.powerUpGraphic.setVisible(state.powerUp.active);
    this.timerText.setText(`Time: ${Math.max(0, Math.ceil((state.endsAt - Date.now()) / 1000))}`);

    const sorted = [...state.players].sort((a, b) => b.score - a.score);
    this.scoreboardText.setText(`Scores\n${sorted.map((p) => `${p.name}${p.isBot ? ' [BOT]' : ''}: ${p.score}`).join('\n')}`);

    if (me) {
      this.myStats.setText(
        `Class: ${me.className}\nRank: ${me.rank}/99\nKills: ${me.totalKills}\nWins: ${me.wins}\nElement CD: ${Math.max(0, ((5000 - (Date.now() - me.lastSpecialAt)) / 1000)).toFixed(1)}s`
      );
    }

    this.elementGraphics.forEach((g) => g.destroy());
    this.elementGraphics = state.elementZones.map((z) => {
      const zone = this.add.circle(z.x, z.y, z.radius, z.color, 0.22);
      const label = this.add.text(z.x, z.y - z.radius - 12, z.label, { fontSize: '12px', color: '#ffffff' }).setOrigin(0.5);
      this.elementGraphics.push(label);
      return zone;
    });

    state.players.forEach((p) => {
      let sprite = this.playerGraphics.get(p.id);
      if (!sprite) {
        const body = this.add.circle(p.x, p.y, 16, p.appearance.tint || p.classMeta.color, 1);
        const accent = this.add.circle(p.x, p.y, 8, p.classMeta.accent, 1);
        const hpBg = this.add.rectangle(p.x, p.y - 28, 42, 6, 0x000000, 0.85);
        const hpBar = this.add.rectangle(p.x - 20, p.y - 28, 40, 4, 0x4dff4d, 1).setOrigin(0, 0.5);
        const name = this.add.text(p.x, p.y - 48, p.name, { fontSize: '14px', color: '#ffffff' }).setOrigin(0.5);
        const aura = this.add.circle(p.x, p.y, 20, 0xf7d25c, 0.1);
        sprite = { body, accent, hpBg, hpBar, name, aura };
        this.playerGraphics.set(p.id, sprite);
      }

      sprite.body.setPosition(p.x, p.y);
      sprite.accent.setPosition(p.x, p.y);
      sprite.hpBg.setPosition(p.x, p.y - 28);
      sprite.hpBar.setPosition(p.x - 20, p.y - 28);
      sprite.name.setPosition(p.x, p.y - 48);
      sprite.aura.setPosition(p.x, p.y);

      sprite.body.setFillStyle(p.appearance.tint || p.classMeta.color, p.alive ? 1 : 0.3);
      sprite.accent.setFillStyle(p.classMeta.accent, p.alive ? 1 : 0.3);
      sprite.hpBar.width = Math.max(0, (p.hp / 100) * 40);
      sprite.hpBar.setFillStyle(p.boostUntil > Date.now() ? 0xf97dff : 0x4dff4d, 1);
      sprite.name.setText(`${p.name}${p.isBot ? ' [BOT]' : ''} Lv${p.rank}`);
      sprite.aura.setAlpha((p.appearance.winAuraLevel || 0) * 0.03);

      if (p.appearance.cosmetic === 'stripe') {
        sprite.accent.setRadius(6);
      } else if (p.appearance.cosmetic === 'spikes') {
        sprite.accent.setRadius(10);
      } else {
        sprite.accent.setRadius(4);
      }

      const visible = p.alive;
      sprite.body.setVisible(visible);
      sprite.accent.setVisible(visible);
      sprite.hpBg.setVisible(visible);
      sprite.hpBar.setVisible(visible);
      sprite.name.setAlpha(visible ? 1 : 0.45);
      sprite.aura.setVisible(visible);
    });
  }
}

class ResultsScene extends Phaser.Scene {
  constructor() {
    super('ResultsScene');
  }

  create(data) {
    this.add.text(700, 140, 'ROUND OVER', { fontSize: '46px', color: '#ffffff' }).setOrigin(0.5);
    const lines = (data.ranking || []).map((entry, idx) => `${idx + 1}. ${entry.name} (${entry.className}) - ${entry.score} KO | Lv${entry.rank}`);
    this.add.text(700, 340, lines.join('\n'), { fontSize: '28px', color: '#ffd65c', align: 'center' }).setOrigin(0.5);
    const winner = data.ranking?.find((r) => r.id === data.winnerId);
    this.add.text(700, 560, winner ? `Winner: ${winner.name} earned a cosmetic aura upgrade!` : 'No winner', { fontSize: '24px', color: '#b6cdfd' }).setOrigin(0.5);
    this.add.text(700, 620, `Room ${data.roomCode} returned to lobby`, { fontSize: '22px', color: '#ffffff' }).setOrigin(0.5);
    this.add.text(700, 680, 'Press ENTER to return to lobby', { fontSize: '26px', color: '#ffffff' }).setOrigin(0.5);
    this.input.keyboard.once('keydown-ENTER', () => this.scene.start('LobbyScene'));
  }
}

new Phaser.Game({
  type: Phaser.AUTO,
  width: 1400,
  height: 900,
  parent: 'game',
  scene: [TitleScene, MenuScene, LobbyScene, ArenaScene, ResultsScene]
});
