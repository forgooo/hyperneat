import neat
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
import math
import pickle

# ---------- Параметры HyperNEAT ----------
INPUT_SIZE = 28  # 28x28
OUTPUT_SIZE = 10
LINK_THRESHOLD = 0.1   # порог для создания связи (по абсолютному значению веса)
TRAIN_SUBSET = 5000    # для ускорения
TEST_SUBSET = 100
GENERATIONS = 50

# ---------- Загрузка MNIST ----------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32)
x_test = x_test.reshape((x_test.shape[0], -1)).astype(np.float32)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train_small = x_train[:TRAIN_SUBSET]
y_train_small = y_train[:TRAIN_SUBSET]
x_test_small = x_test[:TEST_SUBSET]
y_test_small = y_test[:TEST_SUBSET]

# ---------- Подготовка координат субстрата ----------
# Входы: координаты пикселя в диапазоне [-1, 1]
input_coords = []
for r in range(INPUT_SIZE):
    for c in range(INPUT_SIZE):
        # нормируем в [-1,1]
        x = (c / (INPUT_SIZE - 1)) * 2.0 - 1.0
        y = (r / (INPUT_SIZE - 1)) * 2.0 - 1.0
        input_coords.append((x, y))
assert len(input_coords) == INPUT_SIZE * INPUT_SIZE

# Выходы: разместим по линии (или по сетке); используем 1D равномерно в [-1,1]
output_coords = []
for i in range(OUTPUT_SIZE):
    t = (i / (OUTPUT_SIZE - 1)) * 2.0 - 1.0
    output_coords.append((t, 0.0))


# ---------- Функция, которая по CPPN создаёт матрицу весов input->output ----------
def cppn_to_weights(cppn, input_coords, output_coords, link_threshold=LINK_THRESHOLD):
    n_in = len(input_coords)
    n_out = len(output_coords)
    weights = np.zeros((n_out, n_in), dtype=np.float32)  # weights[j,i]
    for j, outc in enumerate(output_coords):
        x2, y2 = outc
        for i, inc in enumerate(input_coords):
            x1, y1 = inc
            inputs = [x1, y1, x2, y2, 1.0]  # bias
            # CPPN -> два выхода: raw_weight, existence_score (или просто weight + threshold)
            out = cppn.activate(inputs)
            raw_w = out[0]
            # optional second output could be used as existence gate: ex = out[1]
            # ex = out[1]
            # Используем порог по абсолютному значению
            if abs(raw_w) >= link_threshold:
                weights[j, i] = raw_w
            else:
                weights[j, i] = 0.0
    return weights


# ---------- Простая функция предсказания (feedforward без активации скрытых узлов) ----------
def predict_with_weights(weights, x):
    # x: (n_in,) in [0,1] - входные пиксели
    # weights: shape (n_out, n_in)
    logits = weights.dot(x)  # shape (n_out,)
    # softmax
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    return probs


# ---------- Fitness: точность на подмножестве ----------
def evaluate_cppn_fitness(genome, config):
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    weights = cppn_to_weights(cppn, input_coords, output_coords, link_threshold=LINK_THRESHOLD)

    correct = 0
    for xi, yi in zip(x_train_small, y_train_small):
        probs = predict_with_weights(weights, xi)
        pred = np.argmax(probs)
        if pred == yi:
            correct += 1
    accuracy = correct / len(x_train_small)
    return accuracy


def eval_genomes(genomes, config):
    for gid, genome in genomes:
        genome.fitness = evaluate_cppn_fitness(genome, config)
    best_genome = max(genomes, key=lambda g: g[1].fitness)[1]
    print(f"Generation best genome {best_genome.key}: nodes={len(best_genome.nodes)}, "
          f"connections={len(best_genome.connections)}, fitness={best_genome.fitness:.3f}")

# ---------- Настройка NEAT ----------
config_path = "config.txt"  # файл конфигурации (см. выше)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

# ---------- Эволюция ----------
winner = pop.run(eval_genomes, GENERATIONS)

# ---------- Сохранение победителя ----------
with open("hyperneat_cppn_winner.pkl", "wb") as f:
    pickle.dump(winner, f)

print("=== WINNER ID:", winner.key if hasattr(winner, "key") else "genome")

# ---------- Тестирование победителя ----------
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
best_weights = cppn_to_weights(winner_net, input_coords, output_coords, link_threshold=LINK_THRESHOLD)

correct = 0
for xi, yi in zip(x_test_small, y_test_small):
    probs = predict_with_weights(best_weights, xi)
    pred = np.argmax(probs)
    if pred == yi:
        correct += 1
acc = correct / len(x_test_small)
print(f"Test accuracy (subset): {acc:.4f}")

# Сохраним веса
np.save("hyperneat_best_weights.npy", best_weights)
print("Saved best weights to hyperneat_best_weights.npy")
