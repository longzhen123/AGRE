import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from src.load_base import load_ratings, data_split, load_kg


def draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker, font_size=18):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.xticks(range(1, len(x_list) + 1), x_list, fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.plot(range(1, len(x_list) + 1),
             y_list,
             marker=marker,
             markerfacecolor='None',
             color=color,
             label=label,
             markersize=font_size)

    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.legend(loc='upper right', fontsize=font_size)
    # 关键代码

    # plt.show()
    plt.savefig('./fig/' + file_name, bbox_inches='tight')


def get_dataset_attribute():

    for dataset in ['music', 'book', 'ml', 'yelp']:
        data_dir = './data/' + dataset + '/'
        ratings_np = load_ratings(data_dir)
        item_set = set(ratings_np[:, 1])
        user_set = set(ratings_np[:, 0])

        kg_dict, n_entity, n_relation = load_kg(data_dir)
        n_entity = n_entity
        n_user = len(user_set)
        n_item = len(item_set)
        n_interaction = int(ratings_np.shape[0] / 2)
        data_density = (n_interaction * 100) / (n_user * n_item)

        print(dataset)
        print('#user: %d \t #item: %d \t #entity: %d \t #relation: %d \t #interaction: %d \t data density: %.4f'
              % (n_user, n_item, n_entity, n_relation, n_interaction, data_density))


if __name__ == '__main__':

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8302731308256054, 0.8336570569609049, 0.8299017910284668, 0.831402612345212, 0.8345230190464401]
    color = 'r'
    file_name = 'music-n.pdf'
    xlabel = '$n$'
    ylabel = 'AUC'
    label = 'Last.FM'
    marker = 'o'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8287124922140102, 0.8334982488828923, 0.8352923117407132, 0.8362202052458649, 0.841556960863571]
    color = 'r'
    file_name = 'music-d.pdf'
    xlabel = '$d$'
    ylabel = 'AUC'
    label = 'Last.FM'
    marker = 'o'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.9000740435878847, 0.9016644096432493, 0.901919141279333, 0.9009596583908761, 0.9000200908730756]
    color = 'g'
    file_name = 'ml-n.pdf'
    xlabel = '$n$'
    ylabel = 'AUC'
    label = 'Movielens-100K'
    marker = 'v'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8887689682936603, 0.8978318106629225, 0.8999341608905862, 0.9020423330734962, 0.8999304379546559]
    color = 'g'
    file_name = 'ml-d.pdf'
    xlabel = '$d$'
    ylabel = 'AUC'
    label = 'Movielens-100K'
    marker = 'v'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8535105030655654, 0.8588470397435413, 0.8687926245721767, 0.8726393903825657, 0.8730592438527105]
    color = 'orange'
    file_name = 'yelp-n.pdf'
    xlabel = '$n$'
    ylabel = 'AUC'
    label = 'Yelp'
    marker = 's'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8682324513546353, 0.8673966335001984, 0.8705393901530928, 0.8757282586953936, 0.8770796048784386]
    color = 'orange'
    file_name = 'yelp-d.pdf'
    xlabel = '$d$'
    ylabel = 'AUC'
    label = 'Yelp'
    marker = 's'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.7231717159183674, 0.7293573681632654, 0.7293337469387755, 0.7363503673469388, 0.7374243069387756]
    color = 'b'
    file_name = 'book-p.pdf'
    xlabel = '$p$'
    ylabel = 'AUC'
    label = 'Book-Crossing'
    marker = 'x'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.7322973910204083, 0.7353195036734694, 0.7359496685714285, 0.7369876571428572, 0.7325608359183674]
    color = 'b'
    file_name = 'book-d.pdf'
    xlabel = '$d$'
    ylabel = 'AUC'
    label = 'Book-Crossing'
    marker = 'x'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    get_dataset_attribute()

