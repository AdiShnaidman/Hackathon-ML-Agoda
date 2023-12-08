import argparse

from hackathon_code.check_performance_task_1 import task_1
from hackathon_code.task_2 import task_2


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Please insert <File-Path-Task1> <File-Path-Task-2> <Task-To-Run>")

    # Add arguments
    parser.add_argument("arg1", type=str, help="Please enter the data file path for task 1(relative)")
    parser.add_argument("arg2", type=str, help="Please enter the data file path for task 2(relative)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments and assign them to relevant variables
    file_path1 = args.arg1
    file_path2 = args.arg2
    task_1(file_path1)
    task_2(file_path2)

if __name__ == '__main__':
    main()
