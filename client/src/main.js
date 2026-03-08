const socket = io();

const shared = {
  playerId: null,
  room: null,
  latestState: null,
  input: { up: false, down: false, left: false, right: false, dash: false, attack: false, special: false, angle: 0 }
};

class TitleScene extends Phaser.Scene {
  constructor() {
    super('TitleScene');
  }

  create() {
    this.add.text(700, 220, 'ARENA BRAWLER PROTOTYPE', { fontSize: '44px', color: '#ffffff' }).setOrigin(0.5);
    this.add.text(700, 320, '2-4 Players | WASD Move | Space Dash | LMB Melee | RMB Special', { fontSize: '22px', color: '#d4dcff' }).setOrigin(0.5);
    this.add.text(700, 430, 'Press ENTER to continue', { fontSize: '30px', color: '#ffd65c' }).setOrigin(0.5);

    this.input.keyboard.once('keydown-ENTER', () => this.scene.start('MenuScene'));
  }
}

class MenuScene extends Phaser.Scene {
  constructor() {
    super('MenuScene');
  }

  create() {
    this.add.text(700, 170, 'ROOM MENU', { fontSize: '44px', color: '#fff' }).setOrigin(0.5);
    this.add.text(700, 250, 'Press C to Create Room', { fontSize: '30px', color: '#77ff77' }).setOrigin(0.5);
    this.add.text(700, 310, 'Press J to Join Room', { fontSize: '30px', color: '#77ccff' }).setOrigin(0.5);
    this.add.text(700, 390, 'Host can start match from lobby with S', { fontSize: '24px', color: '#f7f7f7' }).setOrigin(0.5);

    this.input.keyboard.on('keydown-C', () => {
      const name = window.prompt('Player name?', 'Player') || 'Player';
      socket.emit('createRoom', { name }, (response) => {
        if (!response.ok) return;
        shared.playerId = response.playerId;
        shared.room = response.room;
        this.scene.start('LobbyScene');
      });
    });

    this.input.keyboard.on('keydown-J', () => {
      const roomCode = (window.prompt('Room code?') || '').trim().toUpperCase();
      if (!roomCode) return;
      const name = window.prompt('Player name?', 'Player') || 'Player';
      socket.emit('joinRoom', { roomCode, name }, (response) => {
        if (!response.ok) {
          window.alert(response.error);
          return;
        }
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
    this.title = this.add.text(700, 120, 'Lobby', { fontSize: '42px', color: '#ffffff' }).setOrigin(0.5);
    this.roomCodeText = this.add.text(700, 190, '', { fontSize: '28px', color: '#ffd65c' }).setOrigin(0.5);
    this.playerListText = this.add.text(700, 330, '', { fontSize: '28px', color: '#efefef', align: 'center' }).setOrigin(0.5);
    this.help = this.add.text(700, 630, 'Host presses S to start when 2+ players are in lobby.', { fontSize: '24px', color: '#9fc2ff' }).setOrigin(0.5);

    this.input.keyboard.on('keydown-S', () => {
      socket.emit('startMatch', {}, (response) => {
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
    const lines = roomState.players.map((p) => `${p.id === roomState.hostSocketId ? '[HOST] ' : ''}${p.name}`).join('\n');
    this.playerListText.setText(lines || 'Waiting for players...');
  }
}

class ArenaScene extends Phaser.Scene {
  constructor() {
    super('ArenaScene');
  }

  create() {
    this.playerGraphics = new Map();

    this.add.rectangle(700, 450, 1400, 900, 0x1e2433, 1);
    this.hazardCircle = this.add.circle(0, 0, 90, 0xff4956, 0.35);
    this.powerUpGraphic = this.add.rectangle(0, 0, 26, 26, 0x60ff95, 0.95);

    this.timerText = this.add.text(24, 16, 'Time: 90', { fontSize: '28px', color: '#ffffff' });
    this.scoreboardText = this.add.text(24, 52, '', { fontSize: '20px', color: '#ffffff' });

    this.keys = this.input.keyboard.addKeys('W,A,S,D,SPACE');

    this.input.on('pointerdown', (pointer) => {
      if (pointer.rightButtonDown()) {
        shared.input.special = true;
      } else {
        shared.input.attack = true;
      }
    });

    this.input.mouse?.disableContextMenu();

    socket.off('state');
    socket.on('state', (state) => {
      shared.latestState = state;
      if (state.phase === 'lobby') {
        this.scene.start('LobbyScene');
      }
    });

    socket.off('matchEnded');
    socket.on('matchEnded', ({ ranking, roomCode }) => {
      this.scene.start('ResultsScene', { ranking, roomCode });
    });
  }

  update() {
    const state = shared.latestState;
    if (!state) return;

    // Networking: pack only control intent and let the server run authoritative combat/state.
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

    this.renderState(state);
  }

  renderState(state) {
    this.hazardCircle.setPosition(state.hazard.x, state.hazard.y);
    this.powerUpGraphic.setPosition(state.powerUp.x, state.powerUp.y);
    this.powerUpGraphic.setVisible(state.powerUp.active);

    const timeRemaining = Math.max(0, Math.ceil((state.endsAt - Date.now()) / 1000));
    this.timerText.setText(`Time: ${timeRemaining}`);

    const sorted = [...state.players].sort((a, b) => b.score - a.score);
    this.scoreboardText.setText(`Scores\n${sorted.map((p) => `${p.name}: ${p.score}`).join('\n')}`);

    state.players.forEach((p) => {
      let sprite = this.playerGraphics.get(p.id);
      if (!sprite) {
        const body = this.add.circle(p.x, p.y, 16, 0x66a3ff, 1);
        const hpBg = this.add.rectangle(p.x, p.y - 28, 42, 6, 0x000000, 0.85);
        const hpBar = this.add.rectangle(p.x - 20, p.y - 28, 40, 4, 0x4dff4d, 1).setOrigin(0, 0.5);
        const name = this.add.text(p.x, p.y - 48, p.name, { fontSize: '14px', color: '#ffffff' }).setOrigin(0.5);
        sprite = { body, hpBg, hpBar, name };
        this.playerGraphics.set(p.id, sprite);
      }

      sprite.body.setPosition(p.x, p.y);
      sprite.body.setFillStyle(p.id === shared.playerId ? 0xffd65c : 0x66a3ff, p.alive ? 1 : 0.3);

      sprite.hpBg.setPosition(p.x, p.y - 28);
      sprite.hpBar.setPosition(p.x - 20, p.y - 28);
      sprite.hpBar.width = Math.max(0, (p.hp / 100) * 40);
      sprite.hpBar.setFillStyle(p.boostUntil > Date.now() ? 0xf97dff : 0x4dff4d, 1);
      sprite.name.setPosition(p.x, p.y - 48);
      sprite.body.setVisible(p.alive);
      sprite.hpBg.setVisible(p.alive);
      sprite.hpBar.setVisible(p.alive);
      sprite.name.setAlpha(p.alive ? 1 : 0.5);
    });
  }
}

class ResultsScene extends Phaser.Scene {
  constructor() {
    super('ResultsScene');
  }

  create(data) {
    this.add.text(700, 150, 'ROUND OVER', { fontSize: '46px', color: '#ffffff' }).setOrigin(0.5);

    const lines = (data.ranking || []).map((entry, idx) => `${idx + 1}. ${entry.name} - ${entry.score} KO`);
    this.add.text(700, 340, lines.join('\n'), { fontSize: '30px', color: '#ffd65c', align: 'center' }).setOrigin(0.5);
    this.add.text(700, 560, `Room ${data.roomCode} returned to lobby`, { fontSize: '24px', color: '#b6cdfd' }).setOrigin(0.5);
    this.add.text(700, 640, 'Press ENTER to return to lobby', { fontSize: '26px', color: '#ffffff' }).setOrigin(0.5);

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
