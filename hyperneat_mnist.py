import neat
import matplotlib.pyplot as plt
import statistics

xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs = [(0,), (1,), (1,), (0,)]

def eval_genomes(genomes, config, generation):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

        # Поощрение расширения до 50-го поколения, дальше сжатие
        if generation <= 50:
            nodes_bonus = len(genome.nodes) * 0.5
            conns_bonus = len(genome.connections) * 0.2
            genome.fitness += (nodes_bonus + conns_bonus)
        else:
            nodes_penalty = len(genome.nodes) * 0.05
            conns_penalty = len(genome.connections) * 0.02
            genome.fitness -= (nodes_penalty + conns_penalty)

def plot_stats(stats, accuracy_history):
    generations = list(range(len(stats.generation_statistics)))
    best_fitness = stats.get_fitness_stat(max)
    mean_fitness = stats.get_fitness_stat(statistics.mean)

    # Количество нейронов у лучших геномов каждого поколения
    best_genomes = stats.most_fit_genomes
    node_counts = [len(genome.nodes) for genome in best_genomes]

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(generations, mean_fitness, label="Mean Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness per Generation")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(generations, node_counts, label="Nodes in Best Genome")
    plt.xlabel("Generation")
    plt.ylabel("Node count")
    plt.title("Nodes per Generation")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(generations, accuracy_history, label="Accuracy (best genome)")
    plt.ylim([0, 1.01])
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("XOR Accuracy per Generation")
    plt.legend()

    plt.tight_layout()
    plt.show()

def run(config_file):
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_file
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    accuracy_history = []
    for generation in range(100):
        # Динамическое изменение вероятностей мутаций
        if generation < 50:
            config.genome_config.node_add_prob = 0.9
            config.genome_config.node_delete_prob = 0.1
            config.genome_config.conn_add_prob = 0.9
            config.genome_config.conn_delete_prob = 0.1
        else:
            config.genome_config.node_add_prob = 0.1
            config.genome_config.node_delete_prob = 0.9
            config.genome_config.conn_add_prob = 0.1
            config.genome_config.conn_delete_prob = 0.9

        p.run(lambda g, c: eval_genomes(g, c, generation), 1)

        # Accuracy за поколение (лучший genome)
        best_genome = stats.most_fit_genomes[-1]
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        correct = 0
        for xi, xo in zip(xor_inputs, xor_outputs):
            y = net.activate(xi)[0]
            predicted = int(y > 0.5)
            if predicted == xo[0]:
                correct += 1
        accuracy = correct / 4.0
        accuracy_history.append(accuracy)

    winner = stats.most_fit_genomes[-1]
    print('\nBest genome:\n{!s}'.format(winner))
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    print("\nBest genome XOR test results:")
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print(f"input={xi}, expected={xo[0]}, got={output[0]:.3f}")

    plot_stats(stats, accuracy_history)

if __name__ == '__main__':
    run('config.txt')
