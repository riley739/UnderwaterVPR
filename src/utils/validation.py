import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import PIL


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, val_loader = None, dataset_name='dataset without name ?'):
        
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        # add references
        faiss_index.add(r_list)

        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))
        # print(_)
        # print(predictions)
        
        
        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            if val_loader and q_idx == 47:
                f, axarr = plt.subplots(3,2)
                ref_image,  path = val_loader.dataset.__getitem__(val_loader.dataset.num_references + q_idx, return_path=True)
                q1,  path = val_loader.dataset.__getitem__(pred[0], return_path=True)
                q2,  path = val_loader.dataset.__getitem__(pred[1], return_path=True)
                q3,  path = val_loader.dataset.__getitem__(pred[2], return_path=True)
                q4,  path = val_loader.dataset.__getitem__(pred[3], return_path=True)

                print(gt[q_idx])
                correct, path = val_loader.dataset.__getitem__(gt[q_idx][0], return_path=True)
                correct.save("/home/rbeh9716/Desktop/OpenVPRLab/gt.bmp")
                ref_image.save("/home/rbeh9716/Desktop/OpenVPRLab/query.bmp")
                q1.save("/home/rbeh9716/Desktop/OpenVPRLab/q1.bmp")
                q2.save("/home/rbeh9716/Desktop/OpenVPRLab/q2.bmp")
                q3.save("/home/rbeh9716/Desktop/OpenVPRLab/q3.bmp")
                q4.save("/home/rbeh9716/Desktop/OpenVPRLab/q4.bmp")

                # print(f"Ground Truth was images {gt[q_idx]}, picked {pred[0]} - {pred[0] in gt[q_idx]}")
                # axarr[0,0].imshow(ref_image)
                # axarr[0,0].set_title(f"Query: {q_idx}")
                # axarr[0,1].imshow(correct)
                # axarr[0,1].set_title(f"Database: {gt[q_idx][0]}")

                # axarr[1,0].imshow(q1)
                # axarr[1,0].set_title(f"Database: {pred[0]}")

                # axarr[1,1].imshow(q2)
                # axarr[1,1].set_title(f"Database: {pred[1]}")

                # axarr[2,0].imshow(q3)
                # axarr[2,0].set_title(f"Database: {pred[2]}")

                # axarr[2,1].imshow(q4)
                # axarr[2,1].set_title(f"Database: {pred[3]}")
                
                
                # plt.show()
                # pass
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print('\n') # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performance on {dataset_name}"))
        
        return d, predictions