package com.idibon.ml.alloy;

import com.idibon.ml.predict.PredictModel;
import com.idibon.ml.predict.PredictOptions;
import com.idibon.ml.predict.PredictResult;
import org.json4s.JsonAST;

import java.util.Collections;
import java.util.HashMap;
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
        _labelModelMap = Collections.unmodifiableMap(labelModelMap);
        _labelToUUID = Collections.unmodifiableMap(labelToUUID);
    }

    @Override public Map<String, PredictResult> predict(JsonAST.JObject document, PredictOptions options) {
        // TODO: predict over all? or just a single one? or?
        Map<String, PredictResult> results = new HashMap<>();
        for(Map.Entry<String, PredictModel> entry: _labelModelMap.entrySet()) {
            results.put(entry.getKey(), entry.getValue().predict(document, options));
        }
        return results;
    }
}
