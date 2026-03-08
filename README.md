# Phaser + Socket.IO Arena Brawler Prototype

A playable online 2D top-down arena brawler prototype using **Phaser 3** on the client and **Node.js + Socket.IO** on the server.

## Features
- Top-down 2D arena with placeholder shapes/sprites
- Online room system (host/create + join by room code)
- 2 to 4 player matches
- Per-player actions:
  - movement (WASD)
  - dash (Space)
  - melee attack (left click)
  - cooldown-based special cone attack (right click)
  - health bar
- Arena gameplay objects:
  - hazard zone (damage-over-time circle)
  - power-up pickup (temporary movement + special bonus)
- Match loop:
  - 90-second rounds
  - elimination-based scoring
  - winner is highest score when timer ends
- Screens:
  - title screen
  - room create/join menu
  - lobby
  - in-match HUD
  - results screen

---

## Project Structure

```text
OliTunes/
├── client/
│   ├── index.html          # Phaser bootstrapping + script loading
│   └── src/
│       └── main.js         # Scenes, HUD rendering, input + network client events
├── server/
│   ├── package.json        # Server dependencies and start script
│   └── src/
│       └── index.js        # Room lifecycle, game simulation, Socket.IO events
└── README.md
```

### Why this structure is easy to expand
- **Client and server are fully separated** so you can iterate gameplay visuals independently of backend authority.
- Server simulation is centralized in one tick loop, making it straightforward to add new weapons, hazards, and status effects.
- Phaser scene flow (Title → Menu → Lobby → Match → Results) keeps UI state transitions explicit.

---

## Setup

### 1) Install server dependencies
```bash
cd server
npm install
```

### 2) Start the server
```bash
npm start
```

Server runs at `http://localhost:3000` and serves both Socket.IO and the Phaser client.

### 3) Open the game
Visit:

`http://localhost:3000`

Open multiple browser tabs/windows (or separate devices on the same network tunneling setup) to test multiplayer.

---

## Controls
- `W A S D`: Move
- `Space`: Dash (cooldown)
- `Left Click`: Melee attack
- `Right Click`: Special cone attack (cooldown)

Menu:
- `Enter`: Title -> Menu
- `C`: Create room
- `J`: Join room
- `S`: Start match (host only, lobby only, requires at least 2 players)

Results:
- `Enter`: Return to lobby

---

## Networking Approach (brief)

The prototype uses a **server-authoritative simulation**:
1. Clients send only **input intent** (`up/down/left/right`, attack/dash/special triggers, aim angle).
2. Server runs movement, dash, attacks, hazard damage, respawn, score, timer, and power-up logic.
3. Server broadcasts frequent room snapshots (`state`) to all players.
4. Client renders the latest authoritative snapshot and HUD.

This keeps multiplayer functional and fair while still being responsive enough for prototype combat.

---

## Notes
- Graphics are intentionally placeholder geometry (circles/rectangles).
- No interpolation/prediction is added yet; this is kept simple for expandability.
- Next expansion suggestions:
  - client-side interpolation/prediction
  - multiple maps/hazard patterns
  - additional special abilities and status effects
  - reconnect handling and persistent profiles
