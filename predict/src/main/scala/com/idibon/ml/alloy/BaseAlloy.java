package com.idibon.ml.alloy;

import com.idibon.ml.predict.Document;
import com.idibon.ml.predict.PredictModel;
import com.idibon.ml.predict.PredictOptions;
import com.idibon.ml.predict.PredictResult;
import com.idibon.ml.predict.ensemble.GangModel;
import org.json4s.JsonAST;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Alloy base class.
 *
 * Basically encompasses everything to run predictions for a task.
 */
public abstract class BaseAlloy implements Alloy {

    private static final Logger LOGGER = LoggerFactory.getLogger(BaseAlloy.class);

    private final Map<String, PredictModel> _labelModelMap;

    private final Map<String, String> _labelToUUID;

    public BaseAlloy(Map<String, PredictModel> labelModelMap, Map<String, String> labelToUUID) {
        _labelModelMap = Collections.unmodifiableMap(labelModelMap);
        _labelToUUID = Collections.unmodifiableMap(labelToUUID);
    }

    @Override public Map<String, PredictResult> predict(JsonAST.JObject json, PredictOptions options) {
        // TODO: predict over all? or just a single one? or?
        final Document document = Document.document(json);
        Map<String, PredictResult> results = new HashMap<>();
        for(Map.Entry<String, PredictModel> entry: _labelModelMap.entrySet()) {
            String name = entry.getKey().equals(GangModel.MULTI_CLASS_LABEL()) ? "--multiclass model--" : entry.getKey();
            LOGGER.trace("Predicting for " + name);
            PredictResult prediction = entry.getValue().predict(document, options);
            for(PredictResult result: prediction.getAllResults()){
                LOGGER.trace(result.getLabel() + " " + result.toString());
                results.put(result.getLabel(), result);
            }
        }
        return results;
    }
}
