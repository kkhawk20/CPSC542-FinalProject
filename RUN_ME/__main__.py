# This will run my code!
# in terminal, type in: python3 -m RUN_ME


from . import *
import time

def main():
    print("Running ASL Translation model!l!! ...")
    start_time = time.time()

    train_ds, val_ds, tuner = data_preprocess()
    
    # Create the model
    history = ASL_model(train_ds, val_ds, tuner, retrain = False)

    # Assess model
    assess_model(history)
    
    # For fun, to see how long this thing takes to run :)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

if __name__ == '__main__':
    main()

    


