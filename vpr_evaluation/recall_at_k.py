import matplotlib.pyplot as plt

def plot_recall_at_k(k_values, recall_values_list, labels):
    plt.figure(figsize=(8, 5))
    
    percentages = []
    for vals  in recall_values_list:
        method_percent = []
        for val in vals:
            method_percent.append(val*100)
        percentages.append(method_percent)

    print(percentages)
    for recall_values, label in zip(percentages, labels):
        plt.plot(k_values, recall_values, marker='o', linestyle='-', label=label)

    plt.title('Recall@k - Eiffel')
    plt.xlabel('k')
    plt.ylabel('Recall (%)')

    plt.xticks(k_values)
    plt.ylim(0, 100)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("Eiffel.png")

# Example usage
methods = [
]
k_values = [1, 5, 10, 20]


recall_values_list = [ 
    [0.21782178 ,0.4950495  ,0.68316832 ,0.84158416],
    [0.16831683 ,0.55445545 ,0.75247525 ,0.84059406],
    [0.27722772 ,0.61386139 ,0.78217822 ,0.91089109], #Salad
    [0.37623762 ,0.65346535 ,0.77301254 ,0.86138614],
    [0.37623762 ,0.7029703  ,0.81188119 ,0.91089109], # Boq
    [0.38613861 ,0.56435644 ,0.78217822 ,0.89108911]  # Anyloc  
]

# recall_values_list = [
#     [0.62901554 ,0.6746114 ,0.72435233 ,0.76373057 ,0.81968912 ,0.84663212 ,0.8611399], 
#     [0.65284974 ,0.71295337 ,0.76062176 ,0.78031088 ,0.83316062  ,0.8507772 ,0.86632124],
#     [0.68290155 ,0.75233161 ,0.79378238 ,0.83108808 ,0.87253886 ,0.8880829 ,0.91398964],
#     [0.72953368 ,0.81554404 ,0.85388601 ,0.88290155 ,0.91813472 ,0.93056995 ,0.94093264], 
#     [0.68082902, 0.74715026, 0.80414508, 0.83626943, 0.88601036, 0.90259067, 0.91502591],
#     [0.71088083 ,0.78445596 ,0.8238342  ,0.85284974 ,0.88601036 ,0.89948187 ,0.91295337]
# ]

labels = ['Cosplace', 'MixVPR' , "Salad" , "MegaLoc", "Boq_Finetuned Ours)", "AnyLoc"]

plot_recall_at_k(k_values, recall_values_list, labels)
