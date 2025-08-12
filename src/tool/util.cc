#include <iostream>
#include <random>
#include <algorithm>
#include "util.h"

const int precison = 6;
bool compare_mat_diff(float* mat_a, float* mat_b, int row_num, int col_num) {
    int total_ele_cnt = row_num * col_num;
    int diff_num = 0;
    float delta = 1.0 / std::pow(10, precison);
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            float a = *(mat_a + i * col_num + j);
            float b = *(mat_b + i * col_num + j);
            if (std::abs(a - b) > delta) {
                delta += 1;
            }
        }
    }
    std::cout << "compare mat diff, total_ele_cnt: " << total_ele_cnt << ", diff_num: " << diff_num << ", rate: " << diff_num / (total_ele_cnt * 1.0) << std::endl;
    if (diff_num == 0) {
        return true;
    }
    return false;
}

float getRandomFloat() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    float random_float = dis(gen);
    return random_float;
}
