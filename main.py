from main_baseline import baseline
from main_aug import augmentation
from main_loss import loss

def main():
    print("Do the baseline")
    baseline()
    print("Do the augmentation")
    augmentation()
    print("Do the loss")
    loss()
    print("END")

if __name__ == '__main__':
    main()
    #48923
