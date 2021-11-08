
import twinky.learning as tl


def run_main():
    train_data = tl.create_train_data()

    model = tl.create_model()

    # model = tl.train_model(model, train_data)

    test_data = tl.process_test_data()

    tl.show_results(test_data, model)


if __name__ == '__main__':
    run_main()
