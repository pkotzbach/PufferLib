#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

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
#define GAME_OVER_PENALTY -1.0f
#define CHECK_REWARD 0.25f
#define NO_CHECK_PENALTY -0.02f
#define INVALID_MOVE_PENALTY -0.05f
// #define MOVE_ON_SHOWN_PENTALTY -0.2f


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
} Point;

typedef struct {
    Log log;                        // Required
    char* observations;    // Cheaper in memory if encoded in uint_8
    int* actions;                   // Required
    float* rewards;                 // Required
    unsigned char* terminals;       // Required
    int score;
    int tick;
    int size;
    int bombs;
    char* grid;
    Point cursor;                  // Current cursor position
    float episode_reward;           // Accumulate episode reward
} Game;

void init(Game* game)
{
    game->grid = calloc(game->size * game->size, sizeof(char));
}

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
    for (int i = 0; i < game->size; i++) {
        for (int j = 0; j < game->size; j++) {
            game->observations[i * game->size + j] = game->grid[i * game->size + j];
            //  >= SHOWN ? game->grid[i][j] - SHOWN : EMPTY - 1;
        }
    }
    game->observations[game->size * game->size] = game->cursor.y;
    game->observations[game->size * game->size + 1] = game->cursor.x;
}

void add_log(Game* game) {
    game->log.score = (float)(1 << game->score);
    game->log.perf += ((float)game->score) * REWARD_MULTIPLIER;
    game->log.episode_length += game->tick;
    game->log.episode_return += game->episode_reward;
    game->log.n += 1;
}

void c_reset(Game* game) {
    srand(time(NULL));
    for (int i = 0; i < game->size; i++) {
        for (int j = 0; j < game->size; j++) {
            game->grid[i * game->size + j] = EMPTY;
        }
    }
    game->cursor.x = 0;
    game->cursor.y = 0;
    game->score = 0;
    game->tick = 0;
    game->episode_reward = 0;

    if (game->terminals) game->terminals[0] = 0;

    // Add random bombs
    for (int added = 0; added < game->bombs; ) {
        int pos = rand() % (game->size * game->size);
        int y = pos / game->size;
        int x = pos % game->size;
        if (game->grid[y * game->size + x] != BOMB) {
            game->grid[y * game->size + x] = BOMB;
            // Increment adjacent cells
            for (int i = 0; i < 8; ++i) {
                int nx = x + DX[i], ny = y + DY[i];
                if (nx >= 0 && nx < game->size && ny >= 0 && ny < game->size && game->grid[ny * game->size + nx] != BOMB)
                        game->grid[ny * game->size + nx]++;
            }
            added++;
        }
    }

    update_observations(game);
}

inline void reveal_empty(Game* game, int y, int x)
{
    if (x < 0 || x >= game->size || y < 0 || y >= game->size || game->grid[y * game->size + x] >= SHOWN) return;
    game->grid[y * game->size + x] += SHOWN;
    game->score++;

    if (game->grid[y * game->size + x] == SHOWN) {
        for (int i = 0; i < 8; ++i) {
            int nx = x + DX[i], ny = y + DY[i];
            reveal_empty(game, ny, nx);
        }
    }
}

// void reveal_empty(Game* game, int y, int x) {
//     if (x < 0 || x >= game->size || y < 0 || y >= game->size || game->grid[y][x] >= SHOWN) {
//         return;
//     }

//     Point queue[game->size * game->size];
//     int head = 0;
//     int tail = 0;
//     queue[tail++] = (Point){y, x};
//     game->grid[y][x] += SHOWN;
//     while (head < tail) {
//         Point current = queue[head++];

//         game->score++;
//         if (game->grid[current.y][current.x] != SHOWN) {
//             continue;
//         }

//         for (int i = 0; i < 8; ++i) {
//             int nx = current.x + DX[i];
//             int ny = current.y + DY[i];

//             if (nx >= 0 && nx < game->size && ny >= 0 && ny < game->size && game->grid[ny][nx] < SHOWN) {
//                 game->grid[ny][nx] += SHOWN;
//                 queue[tail++] = (Point){ny, nx};
//             }
//         }
//     }
// }

bool move(Game* game, int move, float* reward) {
    if (move == CHECK && game->grid[game->cursor.y * game->size + game->cursor.x] < SHOWN) {
        if (game->grid[game->cursor.y * game->size + game->cursor.x] == BOMB) return 1;
        reveal_empty(game, game->cursor.y, game->cursor.x);
        *reward = (game->score == 0 ? 3.0f : 1.0f) * CHECK_REWARD;
    }
    else {
        *reward = NO_CHECK_PENALTY;
        if (move == DOWN && game->cursor.y < game->size - 1) game->cursor.y++;
        else if (move == UP && game->cursor.y > 0) game->cursor.y--;
        else if (move == LEFT && game->cursor.x > 0) game->cursor.x--;
        else if (move == RIGHT && game->cursor.x < game->size - 1) game->cursor.x++;
        else *reward = INVALID_MOVE_PENALTY;
    }
    // printf("Move: %d, Cursor: (%d, %d), val: %d, reward: %f\n", move, game->cursor.x, game->cursor.y, game->grid[game->cursor.y][game->cursor.x],*reward);
    return 0;
}

void c_step(Game* game) {
    float reward = 0.0f;
    bool lose = move(game, game->actions[0] + 1, &reward);
    bool win = (game->score == game->size * game->size - game->bombs);
    game->tick++;
    game->terminals[0] = win || lose ? 1 : 0;

    if (lose) reward = GAME_OVER_PENALTY;
    else if (win)
    {
        reward = -1 * GAME_OVER_PENALTY;
    }

    game->rewards[0] = reward;
    game->episode_reward += reward;

    update_observations(game);

    if (game->terminals[0]) {
        add_log(game);
        c_reset(game);
    }
    // printf("reward: %f, score: %d\n", reward, game->score);
    // printf("observations:\n");
    // for (int i = 0; i < game->size * game->size + 2; i++) {
    //     printf("%d ", game->observations[i]);
    // }
}

// Rendering optimizations
void c_render(Game* game) {
    static bool window_initialized = false;
    static const int px = 30;

    if (!window_initialized) {
        InitWindow(px * game->size, px * game->size, "Minesweeper");
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
    for (int i = 0; i < game->size; i++) {
        for (int j = 0; j < game->size; j++) {
            int val = game->grid[i * game->size + j];

            Color color = (Color){60, 60, 60, 255};
            // if (val == BOMB) color = PUFF_RED;
            if (val >= SHOWN && val <= SHOWN + 8) color = PUFF_GREEN;
            if (game->cursor.y == i && game->cursor.x == j) color = PUFF_CYAN;

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
