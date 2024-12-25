
from model_evaluation import evaluate
from data_preprocessing import process_data
from model_training import train

def main():
    process_data()
    train()
    evaluate()

if __name__ == "__main__":
    main()
