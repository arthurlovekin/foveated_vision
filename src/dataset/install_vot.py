import json
import os
import wget,tarfile,zipfile
 
vot_2019_path = '/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/foveated_vision/data'      # object file
json_path = '/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/foveated_vision/json'  # vot 2019 json file
anno_vot = 'vot2019'                           # vot2019 or vot2018 or vot2017
 
 
with open(json_path,'r') as fd:
    vot_2019 = json.load(fd)
home_page = vot_2019['homepage']
 
for i,sequence in enumerate(vot_2019['sequences']):
    print('download the {} sequences'.format(i+1))
    # 
    annotations = sequence['annotations']['url']
    data_url = sequence['channels']['color']['url'].split('../../')[-1]
 
    name = annotations.split('.')[0]
    file_name = annotations.split('.')[0] + '.zip'
 
    down_annotations = os.path.join(home_page,anno_vot,'main',annotations)
    down_data_url = os.path.join(home_page,data_url)
 
 
    image_output_name = os.path.join(vot_2019_path,name,'color',file_name)
    anno_output_name = os.path.join(vot_2019_path,name,file_name)
    out_dir = os.path.dirname(anno_output_name)
 
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)
 
    # annotations download and unzip and remove it
    wget.download(down_annotations, anno_output_name)
    print('loading {} annotation'.format(name))
    # unzip
    file_zip = zipfile.ZipFile(anno_output_name,'r')
    for file in file_zip.namelist():
        file_zip.extract(file, out_dir)
        print('extract annotation {}/{}'.format(name,file))
    file_zip.close()
    os.remove(anno_output_name)
    print('remove annotation {}.zip'.format(name))
 
    # image download and unzip ad remove it
    out_dir = os.path.dirname(image_output_name)
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)
    wget.download(down_data_url,image_output_name)
    print('loading {} sequence'.format(name))
 
    file_zip = zipfile.ZipFile(image_output_name,'r')
    for file  in file_zip.namelist():
        file_zip.extract(file,out_dir)
        print('extract image {}'.format(file))
    file_zip.close()
    os.remove(image_output_name)
    print('remove image file {}.zip'.format(name))
    print('sequence  {} Completed!'.format(i+1))