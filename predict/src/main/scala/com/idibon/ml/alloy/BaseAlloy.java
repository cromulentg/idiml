package com.idibon.ml.alloy;

import com.idibon.ml.feature.Buildable;
import com.idibon.ml.feature.Builder;
import com.idibon.ml.predict.*;
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
public abstract class BaseAlloy<T extends PredictResult & Buildable<T, Builder<T>>> implements Alloy<T> {

    private static final Logger LOGGER = LoggerFactory.getLogger(BaseAlloy.class);

    private final List<PredictModel<T>> _models;

    private final Map<String, Label> _uuidToLabel;

    private final Map<String, ValidationExamples<T>> _validationExamples;

    public BaseAlloy(List<PredictModel<T>> models, Map<String, Label> uuidToLabel,
        Map<String, ValidationExamples<T>> validationExamples) {
        _models = Collections.unmodifiableList(models);
        _uuidToLabel = Collections.unmodifiableMap(uuidToLabel);
        _validationExamples = validationExamples;
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
