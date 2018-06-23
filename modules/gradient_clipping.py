import torch


class GradientClipping:

    @staticmethod
    def clip_gradient_norm(model):
        made_gradient_norm_based_correction = False

        # What is a good max norm for clipping is an empirical question. But a norm
        # of 15 seems to work nicely for this problem.
        # In the beginning there is a lot of clipping,
        # but within an epoch, the total norm is nearly almost below 15
        # so that  clipping becomes almost unnecessary after the start.
        # This is probably what you want: avoiding instability but not also
        # clipping much more or stronger than necessary, as it slows down learning.
        # A max_norm of 10 also seems to work reasonably well, but worse than 15.
        # On person on Quora wrote
        # https://www.quora.com/How-should-I-set-the-gradient-clipping-value
        # "It’s very empirical. I usually set to 4~6.
        # In tensorflow seq2seq example, it is 5.
        # According to the original paper, the author suggests you could first print
        # out uncliped norm and setting value to 1/10 of the max value can still
        # make the model converge."
        # A max norm of 15 seems to make the learning go faster and yield almost no
        # clipping in the second epoch onwards, which seems ideal.
        max_norm = 10
        # https://www.reddit.com/r/MachineLearning/comments/3n8g28/gradient_clipping_what_are_good_values_to_clip_at/
        # https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
        # norm_type is the p-norm type, a value of 2 means the eucledian norm
        # The higher the number of the norm_type, the higher the influence of the
        # outliers on the total_norm. For norm_type = 1 (= "manhattan distance")
        # all values have linear effect on the total norm.
        norm_type = 2

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/9
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm,
                                                    norm_type)

        if total_norm > max_norm:
            made_gradient_norm_based_correction = True
            # print("Made gradient norm based correction. total norm: " + str(total_norm))

        return made_gradient_norm_based_correction, total_norm

    @staticmethod
    def clip_gradient_value(model):
        # Clipping the gradient value is an alternative to clipping the gradient norm,
        # and seems to be more effective
        # https://pytorch.org/docs/master/_modules/torch/nn/utils/clip_grad.html
        # https://www.reddit.com/r/MachineLearning/comments/3n8g28/gradient_clipping_what_are_good_values_to_clip_at/
        # https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
        grad_clip_value_ = 100
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value_)
