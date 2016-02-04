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
import java.util.List;
import java.util.Map;

import static scala.collection.JavaConversions.seqAsJavaList;

/**
 * Alloy base class.
 *
 * Basically encompasses everything to run predictions for a task.
 */
public abstract class BaseAlloy<T extends PredictResult> implements Alloy<T> {

    private static final Logger LOGGER = LoggerFactory.getLogger(BaseAlloy.class);

    private final List<PredictModel<T>> _models;

    private final Map<String, String> _labelToUUID;

    public BaseAlloy(List<PredictModel<T>> models, Map<String, String> labelToUUID) {
        _models = Collections.unmodifiableList(models);
        _labelToUUID = Collections.unmodifiableMap(labelToUUID);
    }

    @Override public List<T> predict(JsonAST.JObject json, PredictOptions options) {
        final Document document = Document.document(json);
        /* classify the document against all models, concatenating the list of
         * PredictResult objects returned by each */
        return _models.stream()
            .flatMap(m -> seqAsJavaList(m.predict(document, options)).stream())
            .collect(java.util.stream.Collectors.toList());
    }
}
