#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

#define BOMBS 40
#define SIZE 16

#define EMPTY 0
#define SHOWN 10
#define BOMB 9

#define UP 1
#define DOWN 2
#define LEFT 3
#define RIGHT 4
#define CHECK 5

// Precomputed constants
#define REWARD_MULTIPLIER 0.09090909f
#define INVALID_MOVE_PENALTY -0.05f
#define GAME_OVER_PENALTY -1.0f
#define GOOD_MOVE_REWARD 0.1f


typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int x;
    int y;
} Cursor;

typedef struct {
    Log log;                        // Required
    unsigned char* observations;    // Cheaper in memory if encoded in uint_8
    int* actions;                   // Required
    float* rewards;                 // Required
    unsigned char* terminals;       // Required
    int score;
    int tick;
    char grid[SIZE][SIZE];
    Cursor cursor;                  // Current cursor position
    float episode_reward;           // Accumulate episode reward
} Game;

// Precomputed color table for rendering optimization
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_GREEN = (Color){0, 187, 0, 255};

// Direction vectors
const int DX[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int DY[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

// --- Logging ---
void add_log(Game* game);

// --- Required functions for env_binding.h ---
void c_reset(Game* env);
void c_step(Game* env);
void c_render(Game* env);
void c_close(Game* env);

// Inline function for updating observations (avoid function call overhead)
static inline void update_observations(Game* game) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            game->observations[i * SIZE + j] = game->grid[i][j];
        }
    }
}

void add_log(Game* game) {
    game->log.score = (float)(1 << game->score);
    game->log.perf += ((float)game->score) * REWARD_MULTIPLIER;
    game->log.episode_length += game->tick;
    game->log.episode_return += game->episode_reward;
    game->log.n += 1;
}

static inline unsigned char calc_score(Game* game) {
    unsigned int revealed = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (game->grid[i][j] > SHOWN) revealed++;
        }
    }
    return revealed;
}

void c_reset(Game* game) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            game->grid[i][j] = EMPTY;
        }
    }

    game->score = 0;
    game->tick = 0;
    game->episode_reward = 0;

    if (game->terminals) game->terminals[0] = 0;

    // Add random bombs
    for (int added = 0; added < BOMBS; ) {
        int pos = rand() % (SIZE * SIZE);
        int y = pos / SIZE;
        int x = pos % SIZE;
        if (game->grid[y][x] != BOMB) {
            game->grid[y][x] = BOMB;
            // Increment adjacent cells
            for (int i = 0; i < 8; ++i) {
                int nx = x + DX[i], ny = y + DY[i];
                if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && game->grid[ny][nx] != BOMB)
                        game->grid[ny][nx]++;
            }
            added++;
        }
    }

    update_observations(game);
}

void reveal_empty(Game* game, int y, int x)
{
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE || game->grid[y][x] >= SHOWN) return;
    if (game->grid[y][x] < SHOWN) game->grid[y][x] += SHOWN;
    if (game->grid[y][x] == SHOWN) {
        for (int i = 0; i < 8; ++i) {
            int nx = x + DX[i], ny = y + DY[i];
            reveal_empty(game, ny, nx);
        }
    }
}

bool move(Game* game, int move, float* reward) {
    bool moved = true;
    // printf("Move: %d, Cursor: (%d, %d), val: %d\n", move, game->cursor.x, game->cursor.y, game->grid[game->cursor.y][game->cursor.x]);
    if (move == DOWN && game->cursor.y < SIZE - 1) game->cursor.y++;
    else if (move == UP && game->cursor.y > 0) game->cursor.y--;
    else if (move == LEFT && game->cursor.x > 0) game->cursor.x--;
    else if (move == RIGHT && game->cursor.x < SIZE - 1) game->cursor.x++;
    else if (move == CHECK) {
            if (game->grid[game->cursor.y][game->cursor.x] == BOMB) return 1;
            reveal_empty(game, game->cursor.y, game->cursor.x);
    }
    else moved = false;

    *reward = moved? GOOD_MOVE_REWARD: INVALID_MOVE_PENALTY;

    return 0;
}

void c_step(Game* game) {
    float reward = 0.0f;
    bool game_over = move(game, game->actions[0] + 1, &reward);
    game->tick++;
    game->terminals[0] = game_over ? 1 : 0;

    if (game_over) {
        reward = GAME_OVER_PENALTY;
        game->score = calc_score(game);
    }

    game->rewards[0] = reward;
    game->episode_reward += reward;

    update_observations(game);

    if (game->terminals[0]) {
        add_log(game);
        c_reset(game);
    }
}

// Rendering optimizations
void c_render(Game* game) {
    static bool window_initialized = false;
    static const int px = 30;

    if (!window_initialized) {
        InitWindow(px * SIZE, px * SIZE + 50, "Minesweeper");
        SetTargetFPS(30); // Increased for smoother rendering
        window_initialized = true;
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Draw grid
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int val = game->grid[i][j];

            Color color = (Color){60, 60, 60, 255};
            // if (val == BOMB) color = PUFF_RED;
            if (game->cursor.y == i && game->cursor.x == j) color = PUFF_CYAN;
            if (val >= SHOWN && val <= SHOWN + 8) color = PUFF_GREEN;

            DrawRectangle(j * px, i * px, px - 5, px - 5, color);
            if (val > SHOWN) {
                char text = (char)((val - SHOWN) + '0');
                DrawText(&text, j * px + 7, i * px + 5, 16, PUFF_WHITE);
            }
        }
    }

    EndDrawing();
}

void c_close(Game* game) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
