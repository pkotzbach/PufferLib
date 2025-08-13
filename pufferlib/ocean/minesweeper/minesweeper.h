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

#define px 30

// Precomputed constants
#define GAME_OVER_PENALTY -1.0f
#define WIN_REWARD 5.0f
#define CHECK_REWARD 0.2f

// actions
#define SKIP 0
#define CHECK 1

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;


typedef struct {
    Log log;                        // Required
    unsigned char* observations;    // Cheaper in memory if encoded in uint_8
    int cursor_idx;
    int observation_edge_size;
    int* actions;                   // Required
    float* rewards;                 // Required
    unsigned char* terminals;       // Required
    int score;
    int tick;
    int size;
    int bombs;
    char* grid;
    float episode_reward;           // Accumulate episode reward
    // UI state
    float message_start_time;
    int message_type; // 0=none, 1=success, 2=failure
    bool show_message;
} Game;

void init(Game* game)
{
    game->grid = calloc(game->size * game->size, sizeof(char));
    game->message_type = 0;
    game->message_start_time = 0.0f;
    game->show_message = false;
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

inline void update_observations(Game* game) {
    // pick random cell
    int random_cell = rand() % (game->size * game->size - game->bombs - game->score);
    int idx = 0;
    for (int i = 0; i < game->size * game->size; i++) {
        if (game->grid[i] < BOMB) idx++;
        if (idx == random_cell) break;
    }

    game->cursor_idx = idx;

    int x = idx % game->size;
    int y = idx / game->size;

    for (int i = 0; i < game->observation_edge_size; i++) {
        for (int j = 0; j < game->observation_edge_size; j++) {
            int nx = x + j - game->observation_edge_size / 2;
            int ny = y + i - game->observation_edge_size / 2;
            int idnx = ny * game->size + nx;
            int idx = i * game->size + j;
            if (nx < 0 || nx >= game->size || ny < 0 || ny >= game->size) {
                game->observations[idx] = SHOWN;
            }
            else {
                game->observations[idx] = game->grid[idnx] >= SHOWN ? game->grid[idnx] - SHOWN + 1 : EMPTY;
            }
        }
    }
}

void add_log(Game* game) {
    game->log.score = (float)(game->score);
    game->log.perf += ((float)game->score) / game->tick;
    game->log.episode_length += game->tick;
    game->log.episode_return += game->episode_reward;
    game->log.n += 1;
}

void c_reset(Game* game) {
    for (int i = 0; i < game->size; i++) {
        for (int j = 0; j < game->size; j++) {
            game->grid[i * game->size + j] = EMPTY;
        }
    }
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

void reveal_empty(Game* game, int start)
{
    int size = game->size;
    if (game->grid[start] >= SHOWN) return;
    if (game->grid[start] != EMPTY) { // reveal a number - no spread
        game->grid[start] += SHOWN;
        game->score++;
        return;
    }

    unsigned char stack[size*size];
    int sp = 0;

    game->grid[start] += SHOWN;
    game->score++;
    stack[sp++] = start;

    while (sp > 0) {
        int idx = stack[--sp];
        int y = idx / size;
        int x = idx - y * size;

        for (int i = 0; i < 8; ++i) {
            int nx = x + DX[i], ny = y + DY[i];
            int nidx = ny * size + nx;
            if (nx < 0 || nx >= size || ny < 0 || ny >= size || game->grid[nidx] >= SHOWN) continue;
            game->grid[nidx] += SHOWN;
            game->score++;
            if (game->grid[nidx] == SHOWN) {
                stack[sp++] = nidx;
            }
        }
    }
}

bool move(Game* game, int move, float* reward) {
    if (move == CHECK) {
        if (game->grid[game->cursor_idx] == BOMB) return 1;
        reveal_empty(game, game->cursor_idx);
        *reward = CHECK_REWARD;
    }
    return 0;
}

void c_step(Game* game) {
    float reward = 0.0f;
    bool lose = move(game, game->actions[0], &reward);
    bool win = (game->score == game->size * game->size - game->bombs);
    game->tick++;
    game->terminals[0] = win || lose ? 1 : 0;

    if (lose)
    {
        reward = GAME_OVER_PENALTY;
        game->message_type = 2;
        game->message_start_time = GetTime();
        game->show_message = true;
    }
    else if (win)
    {
        reward = WIN_REWARD;
        game->message_type = 1;
        game->message_start_time = GetTime();
        game->show_message = true;
    }

    game->rewards[0] = reward;
    game->episode_reward += reward;

    update_observations(game);

    if (game->terminals[0]) {
        // printf("%f\n",game->episode_reward);
        add_log(game);
        c_reset(game);
    }
    // printf("reward: %f, score: %d\n", reward, game->score);
    // printf("observations:\n");
    // for (int i = 0; i < game->size * game->size + 2; i++) {
    //     printf("%d ", game->observations[i]);
    // }
    // usleep(100); // Sleep for 10ms to reduce CPU usage
}

// Rendering optimizations
void c_render(Game* game) {
    static bool window_initialized = false;

    if (!window_initialized) {
        InitWindow(px * game->size, px * game->size + 50, "Minesweeper");
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

            DrawRectangle(j * px, i * px, px - 5, px - 5, color);
            if (val > SHOWN) {
                char text = (char)((val - SHOWN) + '0');
                DrawText(&text, j * px + 7, i * px + 5, 16, PUFF_WHITE);
            }
        }
    }

    if (game->show_message) {
        float currentTime = GetTime();
        float elapsed = currentTime - game->message_start_time;
        if (elapsed < 0.7f) { // Show message for 0.7 seconds
            const char* message = "";
            Color messageColor = PUFF_WHITE;
            if (game->message_type == 1) {
                message = "You won!";
                messageColor = PUFF_CYAN;
            }
            else if (game->message_type == 2) {
                message = "You lost!";
                messageColor = PUFF_RED;
            }
            int fontSize = 20;
            int textWidth = MeasureText(message, fontSize);
            int textX = (GetScreenWidth() - textWidth) / 2;
            int textY = (GetScreenHeight() - fontSize * 2);
            DrawText(message, textX, textY, fontSize, messageColor);
        }
        else {
            game->show_message = false; // Hide message after duration
        }
    }

    EndDrawing();
}

void c_close(Game* game) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    free(game->grid);
}
