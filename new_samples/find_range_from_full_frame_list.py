with open('output_flow_frames.txt','r') as fp:
    video_name = ''
    start_range = 0
    video_frame_extract = ''
    number_extract = 0
    for line in fp:
        video_frame_extract = line[2:line.find('_frame_')]
        if video_name == '':
            video_name = video_frame_extract
            number_extract = int(line[line.find('_frame_') + 7: line.find('.jpg')])
            start_range = number_extract
        elif video_frame_extract != video_name:
            print(video_name.strip(), ",", start_range, ",", number_extract)
            number_extract = int(line[line.find('_frame_') + 7: line.find('.jpg')])
            video_name = video_frame_extract
            start_range = number_extract
        else:
            number_extract = int(line[line.find('_frame_') + 7: line.find('.jpg')])

    print(video_name, ",", start_range, ",", number_extract)
