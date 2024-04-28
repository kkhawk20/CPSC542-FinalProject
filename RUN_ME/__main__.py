# This will run my code!
# in terminal, type in: python3 -m run_me


from . import *
import time

def main():
    print("Running ASL Translation model!l!! ...")
    start_time = time.time()
    # Load the data using the custom SegmentationDataGenerator
    train_ds, test_ds = data_preprocess()
    
    # Create the model
    ASL_model()
    
    # For fun, to see how long this thing takes to run :)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()

    


