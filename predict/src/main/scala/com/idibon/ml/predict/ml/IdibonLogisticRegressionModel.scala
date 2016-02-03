package com.idibon.ml.predict.ml


import java.io.{DataInputStream, DataOutputStream, IOException}

import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.alloy.Codec
import com.idibon.ml.predict.{PredictOptions, SingleLabelDocumentResultBuilder, PredictResult}
import com.idibon.ml.feature.{FeaturePipelineLoader, FeaturePipeline}
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.ml.classification.IdibonSparkLogisticRegressionModelWrapper
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.json4s._

/**
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  *
  * This class implements our LogisticRegressionModel.
  */
case class IdibonLogisticRegressionModel(label: String,
                                         lrm: IdibonSparkLogisticRegressionModelWrapper,
                                         featurePipeline: FeaturePipeline) extends MLModel
  with Archivable[IdibonLogisticRegressionModel, IdibonLogisticRegressionModelLoader] with StrictLogging {

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The model needs to handle "featurization" here.
    *
    * @param document the JObject to pull from.
    * @param options Object of predict options.
    * @return
    */
  override def predict(document: JObject,
                       options: PredictOptions): PredictResult = {
    return predict(featurePipeline.apply(document), options)
  }

  /**
    * The method used to predict from a vector of features.
    *
    * @param features Vector of features to use for prediction.
    * @param options Object of predict options.
    * @return
    */
  override def predict(features: Vector,
                       options: PredictOptions): PredictResult = {
    val results: Vector = lrm.predictProbability(features)
    logger.trace(s"$label: input = ${features.toString} output=${results.toString}")
    val builder = new SingleLabelDocumentResultBuilder(this.getType(), label)
    // get the result of 1, the positive class we're interested in. 0 will be 1.0 minus this value.
    // TODO: change this if we move from binary use case.
    builder.setProbability(results.apply(1).toFloat)
    builder.setMatchCount(1)
    if (!options.significantFeatureThreshold.isNaN()) {
      val indexSigFeatures = lrm.getSignificantFeatures(features, options.significantFeatureThreshold)
      val indexToHumanFeatures = featurePipeline.getHumanReadableFeature(indexSigFeatures.map(x => x._1))
      builder.addSignificantFeatures(indexSigFeatures.map(x => (indexToHumanFeatures.get(x._1).get, x._2)).toList)
    }
    builder.build()
  }

  /**
    * Returns the type of model. Perhaps this should be an enum?
    *
    * @return
    */
  override def getType(): String = this.getClass().getName()

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    *
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  override def getFeaturesUsed(): Vector = {
    //For the predict case this is fine I believe, since we should not have 0 valued features.
    lrm.coefficients
  }

  /** Serializes the object within the Alloy
    *
    * Implementations are responsible for persisting any internal state
    * necessary to re-load the object (for example, feature-to-vector
    * index mappings) to the provided Alloy.Writer.
    *
    * Implementations may return a JObject of configuration data
    * to include when re-loading the object.
    *
    * @param writer destination within Alloy for any resources that
    *               must be preserved for this object to be reloadable
    * @return Some[JObject] of configuration data that must be preserved
    *         to reload the object. None if no configuration is needed
    */
  override def save(writer: Writer): Option[JObject] = {
    val coeffs = writer.within("model").resource("coefficients.libsvm")
    IdibonLogisticRegressionModel.writeCodecLibSVM(
      coeffs, this.lrm.intercept, this.lrm.coefficients, this.label)
    coeffs.close()
    //TODO: store other model metadata like training date, etc.
    val featurePipelineMeta = featurePipeline.save(writer.within("featurePipeline"))
    Some(new JObject(List(
      JField("label", JString(this.label)),
      JField("version", JString(IdibonLogisticRegressionModel.FORMAT_VERSION)),
      JField("feature-meta", featurePipelineMeta.getOrElse(JNothing))
    )))
  }
}

object IdibonLogisticRegressionModel extends StrictLogging {

  val FORMAT_VERSION = "0.0.1"

  /**
    * Static method to write our "libsvm" like format to a stream.
    * @param out
    * @param intercept
    * @param coefficients
    * @param uid
    */
  def writeCodecLibSVM(out: DataOutputStream,
                       intercept: Double,
                       coefficients: Vector,
                       uid: String): Unit = {
    logger.info(s"Writing ${coefficients.size} dimensions with " +
      s"${coefficients.numActives} active dimensions with $intercept for $uid")
    // uid
    Codec.String.write(out, uid)
    // intercept
    out.writeDouble(intercept)
    // dimensions
    Codec.VLuint.write(out, coefficients.size)
    // actual non-zero dimensions
    Codec.VLuint.write(out, coefficients.numActives)
    var maxCoefficient = -10000.0
    var minCoefficient = 10000.0
    coefficients.foreachActive{
      case (index, value) =>
        // do I need to worry about 0?
        Codec.VLuint.write(out, index)
        out.writeDouble(value)
        if (value > maxCoefficient) maxCoefficient = value
        if (value < minCoefficient) minCoefficient = value
    }
  }

  /**
    * Static method to read our "libsvm" like format from a stream.
    * @param in
    * @return
    */
  def readCodecLibSVM(in: DataInputStream): (Double, Vector, String) = {
    // uid
    val uid = Codec.String.read(in)
    // intercept
    val intercept = in.readDouble()
    // dimensions
    val dimensions = Codec.VLuint.read(in)
    // non-zero dimensions
    val numCoeffs = Codec.VLuint.read(in)
    val (indices, values) = (0 until numCoeffs).map { _ =>
      (Codec.VLuint.read(in), in.readDouble())
    }.unzip
    logger.info(s"Read $numCoeffs dimensions from $dimensions for $uid with intercept $intercept")
    (intercept, Vectors.sparse(dimensions, indices.toArray, values.toArray), uid)
  }
}

/** Paired loader class for IdibonLogisticRegressionModel instances */
class IdibonLogisticRegressionModelLoader
  extends ArchiveLoader[IdibonLogisticRegressionModel] with StrictLogging {


  /** Reloads the object from the Alloy
    *
    * @param engine Engine that houses spark context.
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.common.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.common.Archivable#save}
    * @return this object
    */
  def load(engine: Engine, reader: Reader, config: Option[JObject]): IdibonLogisticRegressionModel = {
    implicit val formats = DefaultFormats
    val label = (config.get \ "label" ).extract[String]
    val version = (config.get \ "version" ).extract[String]
    version match {
      case IdibonLogisticRegressionModel.FORMAT_VERSION =>
        logger.info(s"Attemping to load version [v. $version] for '$label'")
      case _ => throw new IOException(s"Unable to load, unhandled version [v. $version] for '$label'")
    }
    val coeffs = reader.within("model").resource("coefficients.libsvm")
    val (intercept: Double,
    coefficients: Vector,
    uid: String) = IdibonLogisticRegressionModel.readCodecLibSVM(coeffs)
    coeffs.close()
    val featureMeta = (config.get \ "feature-meta").extract[JObject]
    val featurePipeline = new FeaturePipelineLoader().load(
      engine, reader.within("featurePipeline"), Some(featureMeta))

    new IdibonLogisticRegressionModel(
      label,
      new IdibonSparkLogisticRegressionModelWrapper(uid, coefficients, intercept),
      featurePipeline)
  }
}
