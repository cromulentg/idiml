package com.idibon.ml.alloy;

import com.idibon.ml.predict.PredictModel;

import java.util.Map;

/**
 * Alloy base class.
 *
 * Basically encompasses everything to run predictions for a task.
 */
public abstract class BaseAlloy implements Alloy {

    private final Map<String, PredictModel> _labelModelMap;

    private final Map<String, String> _labelToUUID;

    public BaseAlloy(Map<String, PredictModel> labelModelMap, Map<String, String> labelToUUID) {
        _labelModelMap = labelModelMap;
        _labelToUUID = labelToUUID;
    }

    @Override public Object predict(Object document, Object options) {
        // TODO: predict over all? or just a single one? or?
        return null;
    }
}
