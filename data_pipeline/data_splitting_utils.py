import numpy as np
import pandas as pd

def assign_instance_counts(file_counts, ideal_partition_counts):
    dynamic_counts = [0 for _ in range(len(ideal_partition_counts))]
    #sort the file counts in descending order
    file_counts = sorted(file_counts, reverse=True)
    partitions = [f'partition_{i}' for i in range(len(ideal_partition_counts))]
    #check if there is an empty partition
    df_assigned = pd.DataFrame(columns=["count", *partitions])
    while len(file_counts) > 0:
        #calculate the difference between the ideal partition counts and the current partition counts
        diff = np.array(ideal_partition_counts) - np.array(dynamic_counts)
        
        #find the index of the partition with the largest difference
        partition_index = np.argmax(diff)
        #check if there are empty partitions
        if min(dynamic_counts) == 0:
            partition_index = np.argmin(dynamic_counts)
        
        #find the index of the patient with the largest count
        count_index = np.argmax(file_counts)
        #assign the patient to the partition with the largest difference
        dynamic_counts[partition_index] += file_counts[count_index]
        #add the patient to the partition as a new row
        zero_list = [0 for _ in range(len(partitions))]
        df_assigned.loc[len(df_assigned.index)] = [file_counts[count_index], *zero_list ]
        df_assigned.loc[len(df_assigned.index)-1, partitions[partition_index]] = 1
        #remove the count from the list
        file_counts.pop(count_index)
    #groupby count
    grouped_count_df = df_assigned.groupby('count').sum().reset_index()
    return grouped_count_df.values

    
   
def split_by_instance_count(instance_list, split_ratios):
    """
    Split a list of instances into multiple lists based on the given split ratios.
    
    Args:
    - instance_list (list): List of instances to split.
    - split_ratios (list): List of ratios to split the instances by.
    
    Returns:
    - split_instances (list): List of lists of instances, split based on the given ratios.
    """
    split_instances = [[] for _ in range(len(split_ratios))]
    #create a df and group by the instances
    instance_df = pd.DataFrame(instance_list, columns=['instance'])
    count_instance_df = instance_df['instance'].value_counts().reset_index()
    count_instance_df.columns = ['instance', 'count']
    #calculate the number of instances to split
    total_count = count_instance_df['count'].sum()
    ideal_split_counts = [int(total_count * ratio) for ratio in split_ratios]
    #assign the counts to the partitions
    instance_count_distribution = assign_instance_counts(count_instance_df['count'].values, ideal_split_counts)
    
    for i in instance_count_distribution:
        count = i[0]
        #get the instances with the count from the count_instance_df
        instances = count_instance_df[count_instance_df['count'] == count]['instance'].values
        #assign the instances to the partitions
        n_assignments = i[1:]
        #randomly pick n instances
        for i, n in enumerate(n_assignments):
            chosen_instances = np.random.choice(instances, n, replace=False)
            split_instances[i].extend(chosen_instances)
            #remove the chosen instances
            instances = [instance for instance in instances if instance not in chosen_instances]
    return split_instances

def stratified_instance_split(instance_list, split_ratios, stratify_column):
    """
    Split a list of instances into multiple lists based on the given split ratios and a stratify column.
    
    Args:
    - instance_list (list): List of instances to split.
    - split_ratios (list): List of ratios to split the instances by.
    - stratify_column (list): List of values to stratify the instances by.
    
    Returns:
    - split_instances (list): List of lists of instances, split based on the given ratios.
    """
    #create a df and group by the stratify column
    instance_df = pd.DataFrame({'instance': instance_list, 'stratify_column': stratify_column})
    grouped_df = instance_df.groupby('stratify_column')['instance'].apply(list).reset_index()
    #do the split by instance count for each group so that the stratification is preserved
    split_instances = [[] for _ in split_ratios]
    for group in grouped_df['instance']:
        splits = split_by_instance_count(group, split_ratios)
        for i, split in enumerate(splits):
            split_instances[i].extend(split)
    return split_instances

def test_split_by_instance_count():
    #test the function
    n_instances = 100
    instance_list_unique = [f'instance_{i}' for i in range(n_instances)]
    #create a class for the instances
    class_list = [f'class_{i}' for i in range(15)]
    #match the instances to the classes
    class_instance_list = np.random.choice(class_list, n_instances)
    #create an array of instances and classes
    matched_instances = np.array([instance_list_unique, class_instance_list]).T
    
    print(matched_instances.shape)
    
    instance_list = []
    #add these instances multiple times randomly between 1 and 10 times to a list
    p = [np.random.random() for _ in range(n_instances)]
    
    #make it less likely to have equal counts
    p = np.multiply(p, p)
    p = [i/sum(p) for i in p]
    for i in range(1000):
        instance_list.extend(np.random.choice(matched_instances.shape[0], np.random.randint(1, n_instances), replace=False, p = p))
    instance_list = matched_instances[instance_list]
    split_ratios = [0.2, 0.3, 0.5]
    split_instances = stratified_instance_split(instance_list[:,0], split_ratios, stratify_column=instance_list[:,1])
    #create a df and value counts
    instance_df = pd.DataFrame(instance_list, columns=['instance', 'class'])
    count_instance_df = instance_df['instance'].value_counts().reset_index()

    #check how the instances are distributed relative to the split ratios
    #calculate the ratios
    total_count = count_instance_df['count'].sum()
    #get the counts of the assigned instances
    split_counts = []
    for split_instance in split_instances:
        #get the counts of the instances
        split_count = count_instance_df[count_instance_df['instance'].isin(split_instance)]['count'].sum()
        split_counts.append(split_count)
    #calculate the ratios
    ratios = [count/total_count for count in split_counts]
    #convert the ratios to a numpy array
    ratios = np.array(ratios)
    print('Diff for instance splitting', ratios - split_ratios)
    #calculate a score for the stratification
    #count the number of instances of each class
    class_counts = instance_df['class'].value_counts().reset_index()
    #calc percentage of each class
    class_distribution = class_counts['count'].values / total_count

    #check the distribution of the classes in the splits
    split_class_counts = []
    for split_instance in split_instances:
        split_class_count = instance_df[instance_df['instance'].isin(split_instance)]['class'].value_counts().reset_index()
        split_class_counts.append(split_class_count)
    #calculate the class distribution for each split
    split_class_distribution = []
    for split_class_count in split_class_counts:
        split_class_distribution.append(split_class_count['count'].values / split_class_count['count'].sum())
    #calculate the difference between the class distribution and the split class distribution
    class_diff = np.array([class_distribution - split_class_dist for split_class_dist in split_class_distribution])
    print('Diff for class distribution', class_diff)


#todo adjust the functions so that there are no magic numbers and strings needed e.g. pass the column as a parameter