#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

#define UNREVEALED 0
#define SHOWN 10
#define BOMB 9

#define UP 1
#define DOWN 2
#define LEFT 3
#define RIGHT 4
#define CHECK 5

#define OBS_EDGE_SIZE 5

// Precomputed constants
#define GAME_OVER_PENALTY -2.0f
#define WIN_REWARD 2.0f
#define CHECK_REWARD 0.05f
#define CHECK_FRONTIER_REWARD 0.3f
#define MOVE_TO_REVALED_PENALTY -0.01f
#define INVALID_MOVE_PENALTY -0.1f


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
    bool covered;
    bool is_bomb;
    float bombs_nearby;
    float revealed_nearby;
} Cell;

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
    Cell* grid;
    bool* revealed;
    Point cursor;                  // Current cursor position
    float episode_reward;           // Accumulate episode reward

    float message_start_time;
    int message_type; // 0=none, 1=success, 2=failure
    bool show_message;
} Game;

void init(Game* game)
{
    game->grid = calloc(game->size * game->size, sizeof(Cell));
    game->revealed = calloc(game->size * game->size, sizeof(bool));
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

// Inline function for updating observations (avoid function call overhead)
void update_observations(Game* game) {
    GridObservation observations[OBS_EDGE_SIZE * OBS_EDGE_SIZE] = {0};
    for (int i = 0; i < OBS_EDGE_SIZE; i++) {
        for (int j = 0; j < OBS_EDGE_SIZE; j++) {
            GridObservation *obs = &observations[i * OBS_EDGE_SIZE + j];
            int nx = game->cursor.x + j - OBS_EDGE_SIZE / 2;
            int ny = game->cursor.y + i - OBS_EDGE_SIZE / 2;
            int grid_idx = ny * game->size + nx;
            int obs_idx = i * OBS_EDGE_SIZE + j;
            if (nx < 0 || nx >= game->size || ny < 0 || ny >= game->size) {
                obs->covered = true;
                obs->bombs_nearby = 0.0f;
                obs->revealed_nearby = 0.0f;
            }
            else {
                obs->covered = game->grid[grid_idx] < SHOWN;
                obs->bombs_nearby = game->grid[grid_idx] >= SHOWN? (float)(game->grid[grid_idx] - SHOWN)/8.0f : 0.0f;
                for (int i = 0; i < 8; ++i) {
                    int rx = nx + DX[i], ry = ny + DY[i];
                    if (nx >= 0 && nx < game->size && ny >= 0 && ny < game->size && game->grid[ny * game->size + nx] != BOMB)
                            game->grid[ny * game->size + nx]++;
                }
            }
                obs->revealed_nearby = (game->revealed[grid_idx]) ? 1.0f : 0.0f;
            }
                // game->observations[obs_idx] = game->grid[grid_idx] >= SHOWN ? game->grid[grid_idx] - SHOWN + 1 : UNREVEALED;
                game->observations[obs_idx] = game->grid[grid_idx];
            }
        }
    }

    char* cursor_channel = game->observations + (OBS_EDGE_SIZE * OBS_EDGE_SIZE);
    cursor_channel[0] = game->cursor.y;
    cursor_channel[1] = game->cursor.x;

    char* visited_channel = game->observations + (OBS_EDGE_SIZE * OBS_EDGE_SIZE) + 2;
    for (int i = 0; i < game->size * game->size; i++) {
        visited_channel[i] = game->revealed[i] ? 1 : 0;
    }
}

void add_log(Game* game) {
    game->log.score = (float)(game->score);
    game->log.perf += ((float)game->score) / ((float)game->tick);
    game->log.episode_length += game->tick;
    game->log.episode_return += game->episode_reward;
    game->log.n += 1;
}

void c_reset(Game* game) {
    for (int i = 0; i < game->size; i++) {
        for (int j = 0; j < game->size; j++) {
            game->grid[i * game->size + j] = UNREVEALED;
            game->revealed[i * game->size + j] = false;
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

void reveal_UNREVEALED(Game* game, int y, int x)
{
    if (x < 0 || x >= game->size || y < 0 || y >= game->size || game->grid[y * game->size + x] >= SHOWN) return;
    game->grid[y * game->size + x] += SHOWN;
    game->revealed[y * game->size + x] = true;
    game->score++;

    if (game->grid[y * game->size + x] == SHOWN) {
        for (int i = 0; i < 8; ++i) {
            int nx = x + DX[i], ny = y + DY[i];
            reveal_UNREVEALED(game, ny, nx);
        }
    }
}


bool move(Game* game, int move, float* reward) {
    if (move == CHECK && game->grid[game->cursor.y * game->size + game->cursor.x] < SHOWN) {
        if (game->grid[game->cursor.y * game->size + game->cursor.x] == BOMB) return 1;
        reveal_UNREVEALED(game, game->cursor.y, game->cursor.x);
        *reward = CHECK_REWARD;
        int nearby_revealed = 0;
        for (int i = 0; i < 8; ++i) {
            int nx = game->cursor.x + DX[i], ny = game->cursor.y + DY[i];
            if (nx >= 0 && nx < game->size && ny >= 0 && ny < game->size && game->grid[ny * game->size + nx] >= SHOWN)
                    nearby_revealed++;
        }

        if (nearby_revealed >= 3)
            *reward = CHECK_FRONTIER_REWARD;
    }
    else {
        bool moved = true;
        if (move == DOWN && game->cursor.y < game->size - 1) game->cursor.y++;
        else if (move == UP && game->cursor.y > 0) game->cursor.y--;
        else if (move == LEFT && game->cursor.x > 0) game->cursor.x--;
        else if (move == RIGHT && game->cursor.x < game->size - 1) game->cursor.x++;
        else moved = false;
        if (moved)
        {
            if (game->revealed[game->cursor.y * game->size + game->cursor.x]) *reward = MOVE_TO_REVALED_PENALTY;
        }
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

    if (lose)
    {
        if (game->score >= game->size) reward = GAME_OVER_PENALTY; // no penalty for small scores
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
    static const int px = 30;

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
            if (game->cursor.y == i && game->cursor.x == j) color = PUFF_CYAN;

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
        free(game->grid);
        free(game->revealed);
    }
}
