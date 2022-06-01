# @title Plotting functions
import matplotlib.pyplot as plt

def plot_mnist(data, nPlots=10):
    """
  Plot MNIST-like data

  Args:
    data: torch.tensor
      MNIST like data to be plotted
    nPlots: int
      Number of samples to be plotted aka Number of plots

  Returns:
    Nothing
  """
    plt.figure(figsize=(12, 8))
    for ii in range(nPlots):
        plt.subplot(1, nPlots, ii + 1)
        plt.imshow(data[ii, 0], cmap="gray")
        plt.axis('off')
    plt.tight_layout
    plt.show()


def multi_task_barplot(accs, tasks, t=None):
    """
  Plot accuracy of multiple tasks

  Args:
    accs: list
      List of accuracies per task
    tasks: list
      List of tasks

  Returns:
    Nothing
  """
    nTasks = len(accs)
    plt.bar(range(nTasks), accs, color='k')
    plt.ylabel('Testing Accuracy (%)', size=18)
    plt.xticks(range(nTasks),
               [f"{TN}\nTask {ii + 1}" for ii, TN in enumerate(tasks.keys())],
               size=18)
    plt.title(t)
    plt.show()


def plot_task(data, samples_num):
    """
  Plots task accuracy

  Args:
    data: torch.tensor
      Data of task to be plotted
    samples_num: int
      Number of samples corresponding to data for task

  Returns:
    Nothing
  """
    plt.plot(figsize=(12, 6))
    for ii in range(samples_num):
        plt.subplot(1, samples_num, ii + 1)
        plt.imshow(data[ii][0], cmap="gray")
        plt.axis('off')
    plt.show()
