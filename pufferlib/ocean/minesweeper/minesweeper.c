#include "minesweeper.h"
#include "puffernet.h"

#define SIZE 5

int main() {
    printf("Starting Minesweeper...\n");
    srand(time(NULL));
    Game env;
    env.size = SIZE;
    env.bombs = 5;
    init(&env);
    unsigned char observations[SIZE * SIZE] = {0};
    unsigned char terminals[1] = {0};
    int actions[1] = {0};
    float rewards[1] = {0};

    env.observations = observations;
    env.terminals = terminals;
    env.actions = actions;
    env.rewards = rewards;

    // Weights* weights = load_weights("resources/g2048/g2048_weights.bin", 134917);
    // int logit_sizes[1] = {4};
    // LinearLSTM* net = make_linearlstm(weights, 1, 16, logit_sizes, 1);
    c_reset(&env);
    c_render(&env);

    // Main game loop
    int frame = 0;
    while (!WindowShouldClose()) {
        c_render(&env);
        frame++;

        int action = -1;
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            printf("click\n");
            Vector2 mouse = GetMousePosition();
            for (int i = 0; i < env.size; i++) {
                for (int j = 0; j < env.size; j++) {
                    Rectangle cell = {j * px, i * px, px - 5, px - 5};
                    if (CheckCollisionPointRec(mouse, cell)) action = i * env.size + j;
                }
            }
            for (int i = 0; i < env.size * env.size; i++)
                printf("%d ", env.grid[i]);
            printf("action: %d\n", action);
            env.actions[0] = action;
        } else if (frame % 10 != 0) {
            continue;
        } else {
            // action = 1;
            // for (int i = 0; i < 16; i++) {
            //     net->obs[i] = env.observations[i];
            // }
            // forward_linearlstm(net, net->obs, env.actions);
        }
        if (action >= 0 && action < env.size * env.size)
            c_step(&env);
    }

    c_close(&env);
    printf("Game Over! Final Max Tile: %d\n", env.score);
    return 0;
}
