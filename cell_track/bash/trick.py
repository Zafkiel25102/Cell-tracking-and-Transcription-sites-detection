
import numpy as np
import cv2
import pandas as pd
import os
# import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


def get_initaldf(df,root_path):
    for id in df.loc[:,'cell_id']:
            
        # id = 561

        # current_row = df.loc[df['cell_id'] == id]
        begin_frame = df.loc[df['cell_id'] == id,'begin']
        end_frame = df.loc[df['cell_id'] == id,'end']

        # 读取开始帧的 CSV 文件
        csv_f_begin = pd.read_csv(root_path + r'/_RES/TRA_' + str(int(begin_frame.iloc[0])) + '.csv')

        # 检查文件是否成功读取
        if not csv_f_begin.empty:
            centroid_row = 0
            centroid_col = 0
            centroid_row = csv_f_begin.loc[csv_f_begin['id'] == id]['centroid_row'].iloc[0]
            centroid_col = csv_f_begin.loc[csv_f_begin['id'] == id]['centroid_col'].iloc[0]
            
            # print(centroid_row)
            
            # 更新当前行的值
            df.loc[df['cell_id'] == id,'begin_centroid_row'] = centroid_row
            df.loc[df['cell_id'] == id,'begin_centroid_col'] = centroid_col
        else:
            print("Error: Unable to read CSV file for begin frame.")

        # 读取结束帧的 CSV 文件
        csv_f_end = pd.read_csv(root_path + r'/_RES/TRA_' + str(int(end_frame.iloc[0])) + '.csv')

        # 检查文件是否成功读取
        if not csv_f_end.empty:
            centroid_row = 0
            centroid_col = 0
            centroid_row = csv_f_end.loc[csv_f_end['id'] == id]['centroid_row'].iloc[0]
            centroid_col = csv_f_end.loc[csv_f_end['id'] == id]['centroid_col'].iloc[0]
            
            # print(centroid_row)
            
            # 更新当前行的值
            df.loc[df['cell_id'] == id,'end_centroid_row'] = centroid_row
            df.loc[df['cell_id'] == id,'end_centroid_col'] = centroid_col
        else:
            print("Error: Unable to read CSV file for begin frame.")

        # ...

    return df


def connect_trajectories(df,result, threshold_distance, threshold_frame, min_frame_threshold):
    # 创建一个新的列来存储连接的轨迹 ID
    df['possible_connections'] = [[] for _ in range(len(df))]

    # 迭代处理每一行轨迹数据
    for i, row1 in df.iterrows():
        # 获取起始帧数和位置
        begin_frame1 = row1['begin']
        end_frame1 = row1['end']

        if row1['cell_id'] not in result:
            continue


        if row1['end'] - row1['begin'] < min_frame_threshold or row1['begin'] != 0:
            continue

        # if (end_frame1 - begin_frame1)<min_frame_threshold:
        #     continue

        # #判断是否为起始帧
        # if begin_frame1 != 0:
        #     continue

        position1 = (row1['end_centroid_row'], row1['end_centroid_col'])
        

        # 寻找与当前轨迹相关的其他轨迹c
        for j, row2 in df.iterrows():
            if i != j:  # 如果不是同一条轨迹
                begin_frame2 = row2['begin']
                end_frame2 = row2['end']

                if row2['cell_id'] not in result:
                    continue

                if row2['begin'] - row1['end'] > min_frame_threshold or row2['end'] - row2['begin'] < min_frame_threshold:
                    continue

                #判断是否晚于1轨迹
                # if begin_frame2 < end_frame1:
                #     continue

                # if (end_frame2 - begin_frame2)<min_frame_threshold:
                #     continue


                position2 = (row2['begin_centroid_row'], row2['begin_centroid_col'])
                

                # 计算帧数和位置之间的距离
                frame_distance = abs(end_frame1 - begin_frame2)
                position_distance = ((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)**0.5

                # 判断是否满足连接条件
                if frame_distance <= threshold_frame and position_distance <= threshold_distance:
                    # 记录可能的连接
                    df.at[i, 'possible_connections'].append(row2['cell_id'])
    
    return df

def process_input():

    root_path_1 = entry_root_path.get()
    # root_path = r'E:/A/20231018-SOX2-CELLID-B/9SOX2-cellid-B/'
    threshold_distance_1 = int(entry_threshold_distance.get())
    threshold_frame_1 = int(entry_threshold_frame.get())
    min_frame_threshold_1 = int(entry_min_frame_threshold.get())
        
    
    result_path = root_path_1 + r'/maskOriginal/'
    results = [os.path.join(result_path, f) for f in os.listdir(result_path) if f.endswith('.tif') or f.endswith('.tiff')]
    results.sort()
    track = np.genfromtxt(root_path_1 + r"/_RES/res_track.txt",dtype=[int, int, int, int])  # 将文件中数据加载到data数组里
    result_1 = []
    for i,res in enumerate(results):

        result_1.append(int(results[i].split('/')[-1].split('_')[1].split('.')[0]))
        # print(result[i])

    df_1 = pd.DataFrame(track)
    df_1 = df_1.rename(columns={'f0': 'cell_id', 'f1': 'begin', 'f2': 'end', 'f3': 'parent'})

    df_2 = get_initaldf(df_1,root_path_1)


    df_post = connect_trajectories(df_2,result_1,threshold_distance_1,threshold_frame_1,min_frame_threshold_1)

    df_post.to_csv(root_path_1 + r'/lineage.csv')
    # 处理完毕后显示结果
    result_label.config(text=f"Processing complete. Result saved to {root_path_1}/lineage.csv")
    





# 创建输入界面
root = tk.Tk()
root.title("Read Tracking Result")

# 创建标签和输入框
label_root_path = tk.Label(root, text="Root Path:")
label_root_path.pack()
entry_root_path = tk.Entry(root)
entry_root_path.pack()

label_threshold_distance = tk.Label(root, text="Threshold Distance:")
label_threshold_distance.pack()
entry_threshold_distance = tk.Entry(root)
entry_threshold_distance.pack()

label_threshold_frame = tk.Label(root, text="Threshold Frame:")
label_threshold_frame.pack()
entry_threshold_frame = tk.Entry(root)
entry_threshold_frame.pack()

label_min_frame_threshold = tk.Label(root, text="Minimum Frame Threshold:")
label_min_frame_threshold.pack()
entry_min_frame_threshold = tk.Entry(root)
entry_min_frame_threshold.pack()


# 创建“Process”按钮
button_process = tk.Button(root, text="Process", command=process_input)
button_process.pack()

# 显示结果的标签
result_label = tk.Label(root, text="")
result_label.pack()

# 运行主循环
root.mainloop()