import segment

if __name__ == '__main__':
    Seg = segment.Segmentation()
    # start predict
    input_dir = r''  # Fill in the image path
    save_dir = r''  # fill in save path
    Seg.predict_and_save(input_dir, save_dir)
