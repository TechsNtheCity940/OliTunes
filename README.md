# Arena Brawler Online Prototype (Phaser 3 + Socket.IO)

A playable online top-down arena brawler built with **Phaser 3** (client) and **Node.js + Socket.IO** (server).

## Features
- 2D arena with placeholder visuals
- Room code multiplayer with host/create and join
- Supports 1-4 human players; empty slots auto-filled with bots at match start
- Host-selectable bot difficulty (`easy`, `normal`, `hard`)
- Character creation before joining:
  - name
  - class: `Earth`, `Fire`, `Wind`, `Water`
  - basic cosmetic variant: `stripe`, `spikes`, `halo`
- Class combat kit (balanced damage across classes):
  - melee direct attack
  - elemental AoE secondary attack (5s cooldown)
    - Earth: Earthquake
    - Fire: Flaming Meteor
    - Wind: Tornado
    - Water: Ice Spike
- Per-player dash, health, rank, and win cosmetics
- Hazard and power-up in arena
- 90-second rounds with elimination scoring
- End-of-round winner = highest kills
- Kill rewards:
  - small health restore on elimination
  - rank progress (levels 1 to 99)
- Winner reward:
  - cosmetic aura upgrade per win
- Screen flow:
  - title -> menu -> lobby -> arena HUD -> results

## Folder Structure
```text
OliTunes/
├── client/
│   ├── index.html
│   └── src/
│       └── main.js
├── server/
│   ├── package.json
│   ├── package-lock.json
│   └── src/
│       └── index.js
└── README.md
```

## Setup
```bash
cd server
npm install
npm start
```

Open `http://localhost:3000`.

## Controls
- `W A S D` move
- `Space` dash
- `Left Click` melee attack
- `Right Click` elemental AoE placement

Lobby:
- `S` start match (host)
- `D` set bot difficulty (host)

## Networking Approach
Server-authoritative multiplayer:
1. Client sends control intent only.
2. Server simulates movement, bot AI, combat, hazard, power-up, rank/score/win progression.
3. Server broadcasts room snapshots (`state`) every tick.
4. Clients render the latest authoritative state.

## Starting a new GitHub repo for this project
If you want this prototype in its own repository:
```bash
git init
git add .
git commit -m "Initial commit: arena brawler prototype"
git branch -M main
git remote add origin <your-new-repo-url>
git push -u origin main
```
