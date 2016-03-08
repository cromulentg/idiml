package com.idibon.ml.predict.ml.metrics;

/*
 * @author "Stefan Krawczyk <stefan@idibon.com>" on 2/15/16.
 */

/**
 * A metric type corresponds to a particular metric we could get back.
 *
 * Note: don't change the names of metrics, since this will break
 * backwards compatibility as we serialize the names. You should feel
 * free to reorder though.
 */
public enum MetricTypes {
    /**
     * The over accuracy of our model.
     */
    Accuracy(FloatMetric.class),
    /**
     * Value for the area under the receiver operating charactistic curve.
     */
    AreaUnderROC(FloatMetric.class),
    /**
     * This is reserved for the threshold that results in the Best F1.
     * Used for tuning binary classifiers.
     */
    BestF1Threshold(FloatMetric.class),
    /**
     * Points to create a confusion matrix.
     */
    ConfusionMatrix(ConfusionMatrixMetric.class),
    /**
     * The harmonic mean between precision and recall.
     */
    F1(FloatMetric.class),
    /**
     * Points to plot F1 by decision threshold.
     */
    F1ByThreshold(PointsMetric.class),
    /**
     * The hamming loss. A metric for the multi-label case.
     */
    HammingLoss(FloatMetric.class),
    /**
     * Properties related to fitting of a model.
     */
    HyperparameterProperties(PropertyMetric.class),
    /**
     * Count of data points for that label.
     */
    LabelCount(LabelIntMetric.class),
    /**
     * F1 but for an individual label.
     */
    LabelF1(LabelFloatMetric.class),
    /**
     * False Positive Rate but for an individual label.
     */
    LabelFPR(LabelFloatMetric.class),
    /**
     * Precision but for an individual label.
     */
    LabelPrecision(LabelFloatMetric.class),
    /**
     * Probabilities for this label from the evaluation set.
     * Useful for computing deciles.
     */
    LabelProbabilities(LabelFloatListMetric.class),
    /**
     * Recall but for an individual label.
     */
    LabelRecall(LabelFloatMetric.class),
    /**
     * Learning Curve F1
     */
    LearningCurveF1(PointsMetric.class),
    /**
     * Learning curve F1 for a label.
     */
    LearningCurveLabelF1(LabelPointsMetric.class),
    /**
     * Learning curve Precision for a label.
     */
    LearningCurveLabelPrecision(LabelPointsMetric.class),
    /**
     * Learning curve recall for a label.
     */
    LearningCurveLabelRecall(LabelPointsMetric.class),
    /**
     * The average of individual label F1s in the non-mutually
     * exclusive case.
     */
    MicroF1(FloatMetric.class),
    /**
     * The average of individual label Precision in the non-mutually
     * exclusive case.
     */
    MicroPrecision(FloatMetric.class),
    /**
     * The average of individual label Recall in the non-mutually
     * exclusive case.
     */
    MicroRecall(FloatMetric.class),
    /**
     * Hacky way to encode portion. i.e. if in included in a sequence
     * of metrics, it'll be assumed that the value in `portion` will
     * reflect the training set portion trained on.
     */
    Portion(FloatMetric.class),
    /**
     * The precision as a float. When we call something
     * foo, how accurate are we?
     */
    Precision(FloatMetric.class),
    /**
     * Points to plot precision by decision threshold.
     */
    PrecisionByThreshold(PointsMetric.class),
    /**
     * Points to plot the precision - recall curve.
     * Useful for showing the trade-off between the two.
     */
    PrecisionRecallCurve(PointsMetric.class),
    /**
     * The recall as a float. Out of all things foo,
     * how many of them do we correctly identify?
     */
    Recall(FloatMetric.class),
    /**
     * Points to plot recall by decision threshold.
     */
    RecallByThreshold(PointsMetric.class),
    /**
     * Points to plot the receiver operating characteristic.
     */
    ReceiverOperatingCharacteristic(PointsMetric.class),
    /**
     * How accurate each label is in the non-mutually exclusive case.
     */
    SubsetAccuracy(FloatMetric.class),
    /**
     * F1 weighted by the number of items in
     * each class (i.e. label)
     */
    WeightedF1(FloatMetric.class),
    /**
     * False positive rate weighted by the number of items in
     * each class (i.e. label)
     */
    WeightedFPR(FloatMetric.class),
    /**
     * Precision weighted by the number of items in
     * each class (i.e. label)
     */
    WeightedPrecision(FloatMetric.class),
    /**
     * Recall weighted by the number of items in
     * each class (i.e. label)
     */
    WeightedRecall(FloatMetric.class);

    MetricTypes(Class<? extends RawMetric> dataType) {
        this.dataType = dataType;
    }

    public final Class<? extends RawMetric> dataType;
}
