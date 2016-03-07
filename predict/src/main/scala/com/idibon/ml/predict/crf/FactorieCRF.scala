package com.idibon.ml.predict.crf

import java.io.{InputStream, DataInputStream, DataOutputStream}
import scala.util.Random

import com.idibon.ml.feature.Freezable
import com.idibon.ml.alloy.Codec

import cc.factorie.{infer => i, variable => v, model => m, optimize => o, la}
import cc.factorie.util.{BinarySerializer, TensorListCubbie}

import org.apache.spark.mllib.linalg.Vector

/** A conditional random field model implemented using FACTORIE
  *
  * CRF models assign tags to a sequence of observations. Tags are derived
  * from labels, and observations are derived from the document text.
  */
class FactorieCRF(dimensions: Int) extends Freezable[FactorieCRF] {

  // Abbreviated type identifier for assignable, inferred elements
  private[this] type TaggedVar = v.CategoricalVariable[String] with CRFTag

  // Stores all assignable tags
  private[crf] val tagDomain = new v.CategoricalDomain[String]()

  // Stores all known features
  private[crf] val featureDomain = new v.DiscreteDomain(dimensions)

  // Stores the learned model factors, and templates to index them
  private[crf] val model = new m.TemplateModel with m.Parameters

  /** Assigns a tag from the tagDomain to every entry in the sequence
    *
    * @param sequence list of feature vectors in the sequence
    * @return list of tags and model confidence for each assignment
    */
  def predict(sequence: Traversable[Vector]): Seq[(BIOTag, Double)] = {
    val inferred: Seq[TaggedVar] = sequence.foldLeft(new CRFSequence)(
      (result, vector) => {
        result += new CRFObservation(vector, this); result
      }).links.map(_.tag)

    val summary = i.BP.inferChainMax(inferred, model)
    summary.setToMaximize(null)

    val scores = new Array[Double](inferred.size)

    /* accumulate the factor weights for the assigned tags across the
     * sequence, to compute a value for predictive confidence */
    summary.factorMarginals.foreach(_ match {
      case m: i.MAPSummary#SingletonFactorMarginal => {
        m.variables.foreach(_ match {
          case t: TaggedVar @unchecked => scores(t.obs.position) += m.score
          case _ =>
        })
      }
      case _ =>
    })

    val biases = model.parameters.tensors(1)

    /* convert the inferred CRFTags to BIOTags, and compute per-assignment
     * confidence by treating each tag's bias term as 50% predictive
     * confidence (i.e., an assignment for a single-element sequence where
     * the one element is completely out-of-vocabulary will have 50%
     * confidence. */
    inferred.zip(scores).map({ case (tag, score) => {
      val bias = biases(tag.value.intValue)
      val confidence = 1.0 / (1.0 + Math.exp(bias - score))
      BIOTag(tag.value.category) -> confidence
    }})
  }

  /** Writes a FactorieCRF to an output stream
    *
    * @param stream output stream
    */
  def serialize(stream: DataOutputStream): this.type = {
    import cc.factorie.util.CubbieConversions._
    Codec.VLuint.write(stream, FactorieCRF.VERSION)
    Codec.VLuint.write(stream, dimensions)
    BinarySerializer.serialize(tagDomain, stream)
    BinarySerializer.serialize(model.parameters.tensors, stream)
    this
  }

  /** Freezes the feature and tag domains in this CRF, adds model templates
    *
    * FactorieCRF instances must be frozen prior to training or prediction.
    */
  def freeze(): FactorieCRF.this.type = {
    tagDomain.freeze()
    featureDomain.freeze()

    /* add all of the parameter templates the first time the object is frozen;
     * these values will either be learned during training (in which case a new
     * weights object will be instantiated for each term) or were loaded by
     * a call to FactorieCRF.deserialize */
    if (model.templates.isEmpty) {
      val loaded = model.parameters.keys.nonEmpty
      // add observation p(tag | feature) term
      model += new m.DotTemplateWithStatistics2[TaggedVar, CRFObservation] {
        val weights = loaded match {
          case true => model.parameters.keys(0).asInstanceOf[m.Weights2]
          case _ => model.Weights(new la.DenseTensor2(tagDomain.size, featureDomain.dimensionSize))
        }
        def unroll1(tag: TaggedVar) = Factor(tag, tag.obs)
        def unroll2(obs: CRFObservation) = Factor(obs.tag, obs)
      }
      // add bias p(tag) term
      model += new m.DotTemplateWithStatistics1[TaggedVar] {
        val weights = loaded match {
          case true => model.parameters.keys(1).asInstanceOf[m.Weights1]
          case _ => model.Weights(new la.DenseTensor1(tagDomain.size))
        }
      }
      // add markov p(tag | tag[n-1]) term
      model += new m.DotTemplateWithStatistics2[TaggedVar, TaggedVar] {
        val weights = loaded match {
          case true => model.parameters.keys(2).asInstanceOf[m.Weights2]
          case _ => model.Weights(new la.DenseTensor2(tagDomain.size, tagDomain.size))
        }
        def unroll1(t: TaggedVar) = if (t.hasPrev) Factor(t.prev, t) else Nil
        def unroll2(t: TaggedVar) = if (t.hasNext) Factor(t, t.next) else Nil
      }
    }

    this
  }
}

/** Training mixin for FactorieCRF
  *
  * Adds methods needed for training models on an as-needed, per-instance basis
  * to FactorieCRF.
  */
trait TrainableFactorieModel {

  private[crf] val model: m.TemplateModel with m.Parameters

  /** Converts a list of tagged feature vectors to a list of CRFKnownTag
    *
    * @param obs observed, tagged feature vectors
    * @return list of CRFKnownTag, ready to be used as training data
    */
  def observe(obs: Traversable[(String, Vector)]): Traversable[CRFKnownTag] = {
    val asCRF = this.asInstanceOf[FactorieCRF]
    obs.foldLeft(new CRFSequence)(
      (seq, tagged) => {
        seq += new CRFObservation(tagged._2, asCRF, tagged._1)
        seq
      }).links.map(_.tag.asInstanceOf[CRFKnownTag])
  }

  /** Trains the model parameters using the provided list of known tags
    *
    * @param trainingData list of training items, each a sequence of CRFKnownTag
    * @param prng random number generator
    */
  def train(trainingData: Traversable[Traversable[CRFKnownTag]], prng: Random) {
    this.asInstanceOf[FactorieCRF].freeze()
    val examples = trainingData.map(doc => {
      new o.LikelihoodExample(doc.toIterable, model, i.InferByBPChain)
    })
    o.Trainer.batchTrain(model.parameters, examples.toSeq)(prng)
  }
}

// ====== INTERNAL TYPES FOR CONSTRUCTING CRFs ========
/** Links a CategoricalVariable to a CRFObservation
  *
  * Allows navigating a Chain of CategoricalVariables via the doubly-linked
  * list interface supported by the Chain
  */
trait CRFTag {
  /** The linked CRFObservation that this variable assigns */
  val obs: CRFObservation
  /** The inferred tag for the CRFObservation */
  def value: v.CategoricalValue[String]

  def hasNext = obs.hasNext
  def hasPrev = obs.hasPrev
  def next = obs.next.tag
  def prev = obs.prev.tag
}

object FactorieCRF {
  val VERSION_1 = 1
  val VERSION = VERSION_1

  /** Loads a FactorieCRF from an input stream
    *
    * @param stream input stream
    */
  def deserialize(stream: InputStream): FactorieCRF = {
    import cc.factorie.util.CubbieConversions._

    val ds = stream match {
      case d: DataInputStream => d
      case _ => new DataInputStream(stream)
    }

    Codec.VLuint.read(ds) match {
      case VERSION_1 =>
      case _ => throw new UnsupportedOperationException("Version")
    }

    val dimensions = Codec.VLuint.read(ds)
    val model = new FactorieCRF(dimensions)
    BinarySerializer.deserialize(model.tagDomain, ds)
    /* Factorie's standard save / load mechanism doesn't work very well
     * for template models -- we need to read the tensors directly, insert
     * them into the model parameters list, then add the model templates
     * so that each template can reference the loaded weights. */
    val tensors = new TensorListCubbie[Seq[la.Tensor]]
    BinarySerializer.deserialize(tensors, ds)
    tensors.fetch.foreach(_ match {
      case t: la.Tensor1 => {
        val w = model.model.Weights(new la.SparseTensor1(t.dim1))
        w.set(t)
      }
      case t: la.Tensor2 => {
        val w = model.model.Weights(new la.SparseIndexedTensor2(t.dim1, t.dim2))
        w.set(t)
      }
      case t: la.Tensor3 => {
        val w = model.model.Weights(new la.SparseIndexedTensor3(t.dim1, t.dim2, t.dim3))
        w.set(t)
      }
    })
    model.freeze()
  }
}

/** A tagged variable with a known (i.e., annotated) value
  *
  * Used for training models and evaluating model accuracy by comparing the
  * inferred value (aka "Aimed" within the FACTORIE infrastructure) against the
  * assigned (aka "Target") value.
  *
  * @param obs the observation assigned by this variable
  * @param domain the domain of possible tags
  * @param assigned the assigned tag
  */
case class CRFKnownTag(override val obs: CRFObservation,
  override val domain: v.CategoricalDomain[String], val assigned: String)
    extends v.LabeledCategoricalVariable(assigned) with CRFTag

/** A tagged variable with no known value
  *
  * Used for inference to assign tags from the domain to an unknown sequence
  *
  * @param obs the observation assigned by this variable
  * @param domain the domain of possible tags
  */
case class CRFUnknownTag(override val obs: CRFObservation,
  override val domain: v.CategoricalDomain[String])
    extends v.CategoricalVariable[String] with CRFTag

/** An observed value within a taggable CRFSequence
  *
  * CRFObservations consist of a set of features identified for the inferred
  * element from within the domain of all features, and a tag from the domain
  * of all tags, optionally with a known value.
  *
  * @param domain the CRF feature domain
  * @param tag assigned tag (optionally with known value)
  */
class CRFObservation(val domain: v.VectorDomain)
    extends v.VectorVariable(new la.GrowableSparseBinaryTensor1(domain.dimensionDomain))
    with v.ChainLink[CRFObservation, CRFSequence] {

  def this(features: Vector, crf: FactorieCRF) {
    this(crf.featureDomain)
    _tag = new CRFUnknownTag(self, crf.tagDomain)
    features.foreachActive({ case (i, v) => if (v != 0.0) value.update(i, 1.0) })
  }

  def this(features: Vector, crf: FactorieCRF, assigned: String) {
    this(crf.featureDomain)
    _tag = new CRFKnownTag(this, crf.tagDomain, assigned)
    features.foreachActive({ case (i, v) => if (v != 0.0) value.update(i, 1.0) })
  }

  @inline def tag = _tag

  private[this] var _tag: v.CategoricalVariable[String] with CRFTag = null

  def self = this
}

/** A sequence of CRFObservations that is independently inferencable */
class CRFSequence extends v.Chain[CRFSequence, CRFObservation]
