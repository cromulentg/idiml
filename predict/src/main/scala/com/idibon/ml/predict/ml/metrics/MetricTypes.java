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
    BestF1Threshold(FloatMetric.class),
    AreaUnderROC(FloatMetric.class),
    ReceiverOperatingCharacteristic(PointsMetric.class),
    PrecisionRecallCurve(PointsMetric.class),
    F1ByThreshold(PointsMetric.class),
    PrecisionByThreshold(PointsMetric.class),
    RecallByThreshold(PointsMetric.class),
    Precision(FloatMetric.class),
    Recall(FloatMetric.class),
    F1(FloatMetric.class),
    LabelPrecision(LabelFloatMetric.class),
    LabelRecall(LabelFloatMetric.class),
    LabelF1(LabelFloatMetric.class),
    LabelFPR(LabelFloatMetric.class),
    WeightedPrecision(FloatMetric.class),
    WeightedRecall(FloatMetric.class),
    WeightedF1(FloatMetric.class),
    WeightedFPR(FloatMetric.class),
    ConfusionMatrix(ConfusionMatrixMetric.class),
    LabelCount(LabelIntMetric.class),
    HyperparameterProperties(PropertyMetric.class),
    MicroF1(FloatMetric.class),
    MicroPrecision(FloatMetric.class),
    MicroRecall(FloatMetric.class),
    HammingLoss(FloatMetric.class),
    SubsetAccuracy(FloatMetric.class),
    Accuracy(FloatMetric.class);

    MetricTypes(Class<? extends RawMetric> dataType) {
        this.dataType = dataType;
    }

    public final Class<? extends RawMetric> dataType;
}
