import numpy as np

class Loss:
    def calculate(self, output, y_true):
        sample_losses = self.forward(output, y_true)
        data_loss = np.mean(sample_losses)

        return data_loss

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, -1e-7, +1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum((y_pred_clipped * y_true), axis=1)

        else:
            pass

        negative_loss_likehoods = -np.log(correct_confidences)

        return negative_loss_likehoods

