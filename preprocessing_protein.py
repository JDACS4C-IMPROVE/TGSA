import gzip
import tqdm
import os

filter_threshold = 0.95 * 1000

# 20052394042 lines in total
edge_fn = "./benchmark_dataset_generator/csa_data/raw_data/x_data/protein.links.detailed.v11.5.txt.gz"
# 191105745 after filtering
filtered_edge_fn = "./benchmark_dataset_generator/csa_data/raw_data/x_data/protein.links.detailed.v11.5_filter_95.txt"

if not os.path.exists(filtered_edge_fn):
    fp = open(filtered_edge_fn, 'w')

    count_filtered = 0
    with gzip.open(edge_fn, 'rt') as f:
        line_cnt = 0
        for line in f:
            if line_cnt == 0:
                fp.write(line)
            else:
                ele = line.split(" ")
                combined_score = int(ele[-1].strip())
                if combined_score > filter_threshold:
                    fp.write(line)
                    # print('got line', line)
                else:
                    count_filtered += 1
            line_cnt += 1

            if line_cnt % 100 == 0:
                print("Filter ratio = %f " % (count_filtered/line_cnt))

    fp.close()

###############################################################################
edge_fn = "./benchmark_dataset_generator/csa_data/raw_data/x_data/protein.links.v11.5.txt.gz"
filtered_edge_fn = "./benchmark_dataset_generator/csa_data/raw_data/x_data/protein.links.v11.5_filter_95.txt"
if not os.path.exists(filtered_edge_fn):
    fp = open(filtered_edge_fn, 'w')

    count_filtered = 0
    with gzip.open(edge_fn, 'rt') as f:
        line_cnt = 0
        for line in f:
            if line_cnt == 0:
                fp.write(line)
            else:
                ele = line.split(" ")
                combined_score = int(ele[-1].strip())
                if combined_score > filter_threshold:
                    fp.write(line)
                    # print('got line', line)
                else:
                    count_filtered += 1
            line_cnt += 1

            if line_cnt % 100 == 0:
                print("Filter ratio = %f " % (count_filtered/line_cnt))

    fp.close()


