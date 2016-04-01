package com.idibon.ml.predict.ml

import org.apache.spark.mllib.classification.IdibonSparkMLLIBLRWrapper
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers, ParallelTestExecution}

/**
  * Class to test IdibonSparkMLLIBLRWrapper methods.
  */
class IdibonSparkMLLIBLRWrapperSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution {

  /** Real feature vectors from a real use case **/
  val vectors = Seq(
    (0.0, Vectors.sparse(46, Array(4), Array(1.0))),
    (0.0, Vectors.sparse(46, Array(3,4,8,9,10,11,13,24,26,34,35,36,37,38,39,40,41,42,43,44),
      Array(1.0,2.0,2.0,2.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,2.0,1.0))),
    (0.0, Vectors.sparse(46, Array(1,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,27,28,31,34,35,36,37,38,41),
      Array(1.0,2.0,3.0,2.0,2.0,2.0,3.0,1.0,1.0,2.0,2.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))),
    (1.0, Vectors.sparse(46, Array(4,7,11,14,25), Array(2.0,1.0,1.0,1.0,1.0))),
    (1.0, Vectors.sparse(46, Array(0,1,2,3,10,30), Array(2.0,2.0,2.0,1.0,1.0,1.0))),
    (1.0, Vectors.sparse(46, Array(3,4,8,9,12,24,26,27,28,29,30), Array(1.0,2.0,2.0,4.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0))),
    (2.0, Vectors.sparse(46, Array(3,4,5,8,9,15,22,23,33,42,44), Array(1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))),
    (2.0, Vectors.sparse(46, Array(3,4,8,9,20,30,31,32,33), Array(2.0,2.0,2.0,3.0,1.0,1.0,1.0,2.0,1.0))),
    (2.0, Vectors.sparse(46, Array(4,8,9,11,22,23,24,25,26,40), Array(1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))),
    (2.0, Vectors.sparse(46, Array(20,28,45), Array(1.0,1.0,2.0)))
  )

  /** Real weights that correspond to training on the above feature vectors.
    * We overfit like crazy and thus makes this unit test a little easier to
    * grapple with.
    */
  val weights = Vectors.dense(Array(0.6915846965385272,0.4689644384651143,0.6915846965385272,
  0.19788386826948368,-0.623320903088721,-0.01416517405761701,-0.31497261657318604,0.707364194323377,
  -0.012936043041228968,0.18075743415123272,0.17052786756159427,0.45133272931335333,0.19526443215854414,
  -0.3545596687126197,2.166964941598487,-0.3702800144596527,-0.31497261657318604,-0.31497261657318604,
  -0.31497261657318604,-0.31497261657318604,-0.43142596296605706,-0.31497261657318604,-0.5137859336466726,
  -0.5137859336466726,0.10465894962188968,2.0234590224114735,0.38958675663060593,0.8547093010511051,
  0.4281715610803529,1.0747142175075017,1.373018967289854,-0.539343077378677,-0.16444345220786089,
  -0.2009347045486625,-0.5542149574312514,-0.5542149574312514,-0.5542149574312514,-0.5542149574312514,
  -0.5542149574312514,-0.17766290114348157,-0.6977208766182726,-0.5542149574312514,-0.2158065846012355,
  -0.17766290114348157,-0.2158065846012355,-0.26020811300723784,-0.9642561208662076,-0.1333422561030652,
  -0.17712153352263935,-0.1333422561030652,0.3345706891015085,-1.0831035699935643,0.3903129885410235,
  -0.13675898081765755,-0.1950279927331063,0.5022198220304741,0.13849529665807156,-0.5060384421585239,
  0.011744885962397148,-0.26299622541038176,-0.2606904470778449,-0.3459053779587185,0.2852482586887883,
  -0.13675898081765755,-0.13675898081765755,-0.13675898081765755,-0.13675898081765755,0.813894202273753,
  -0.13675898081765755,1.5386631094311083,1.5386631094311083,0.14611697288731323,0.9075094727836008,
  0.28421280971492136,-0.5200739741942395,0.8135393708657862,-0.3255289962438883,0.33664168056886695,
  0.8042185112170978,0.8516198796773001,1.3971744767456178,-0.514165417217112,-0.514165417217112,
  -0.514165417217112,-0.514165417217112,-0.514165417217112,-0.32027694559755493,0.739249433525208,
  -0.514165417217112,0.07879054831140431,-0.32027694559755493,0.07879054831140431,1.4114172430716316,
  -0.7075983699223728))

  val model = new IdibonSparkMLLIBLRWrapper(
    Vectors.sparse(10, Array(0, 3, 5), Array(0.6, 0.8, 0.1)).toDense, 0.0, 10, 2)

  val multinomialmodel = new IdibonSparkMLLIBLRWrapper(weights, 0.0, 46, 3)

  describe("predict probability binary case") {
    it("handles vectors of different sizes well when we need to prune") {
      val result = model.predictProbability(Vectors.sparse(11, Array(0, 3, 11), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(2, Array(0, 1), Array(0.1978161114414183,0.8021838885585817))
    }

    it("handles vectors of different sizes well") {
      val result = model.predictProbability(Vectors.sparse(11, Array(0, 3, 9), Array(1.0, 1.0, 0.0)))
      result shouldBe Vectors.sparse(2, Array(0, 1), Array(0.1978161114414183,0.8021838885585817))
    }

    it("handles vectors equal to model size") {
      val result = model.predictProbability(Vectors.sparse(10, Array(0, 3, 9), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(2, Array(0, 1), Array(0.1978161114414183,0.8021838885585817))
    }
  }

  describe("predict probability n-ary case") {
    it("should have all predictions sum to 1.0") {
      vectors.foreach({case (_, vector) =>
        val result = multinomialmodel.predictProbability(vector)
        result.toDense.values.sum shouldBe 1.0
      })
    }
    it("should have the max probability class match the internal spark predict method") {
      vectors.foreach({case (gold, vector) =>
        val result = multinomialmodel.predictProbability(vector)
        val maxClass = result.argmax
        val sparkResult = multinomialmodel.predict(vector)
        maxClass shouldBe sparkResult.toInt
        // we should match the gold label provided too!
        maxClass shouldBe gold.toInt
      })
    }

    it("throws assertion error on vector size smaller than model size") {
      intercept[AssertionError] {
        multinomialmodel.predictProbability(Vectors.sparse(11, Array(0, 3, 11), Array(1.0, 1.0, 1.0)))
      }
    }

    it("handles vectors of different sizes well") {
      val result = multinomialmodel.predictProbability(Vectors.sparse(50, Array(0, 3, 11), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(3, Array(0,1,2), Array(0.16503130147671052,0.6307669474888523,0.2042017510344372))
      result.toDense.values.sum shouldBe 1.0
    }

    it("handles vectors equal to model size") {
      val result = multinomialmodel.predictProbability(Vectors.sparse(46, Array(0, 3, 9), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(3, Array(0,1,2), Array(0.356634355451551,0.3965019347305708,0.24686370981787817))
      result.toDense.values.sum shouldBe 1.0
    }
  }

  describe("get significant features") {
    it("handles getting no significant features when all features map to zero weight value") {
      val result = model.getSignificantDimensions(
        Vectors.sparse(10, Array(1, 2, 4), Array(1.0, 1.0, 1.0)), 0.55f)
      result.size shouldBe 2
      result(0) shouldBe (0, Vectors.zeros(10))
      result(1) shouldBe (1, Vectors.zeros(10))
    }
    it("handles getting no significant features when values are below the threshold") {
      val result = model.getSignificantDimensions(
        Vectors.sparse(10, Array(1, 2, 5), Array(1.0, 1.0, 1.0)), 0.90f)
      result.size shouldBe 2
      result(0) shouldBe (0, Vectors.zeros(10))
      result(1) shouldBe (1, Vectors.zeros(10))
    }
    it("handles getting significant features when values are above the threshold") {
      val result = model.getSignificantDimensions(
        Vectors.sparse(10, Array(0, 3, 5), Array(1.0, 1.0, 1.0)), 0.53f)
      result.size shouldBe 2
      result(0) shouldBe (0, Vectors.zeros(10))
      result(1) shouldBe (1, Vectors.sparse(10, Array(0, 3), Array(0.6456563062257954, 0.6899744811276125)))
    }

    it("handles n-ary case of getting significant features") {
      val result = multinomialmodel.getSignificantDimensions(vectors(2)._2, 0.70f)
      result.size shouldBe 3
      result(0) shouldBe (0, Vectors.sparse(46, Array(4), Array(0.7292545321646238)))
      result(1) shouldBe (1, Vectors.sparse(46, Array(14), Array(0.7116800050733418)))
      result(2) shouldBe (2, Vectors.sparse(46, Array(), Array()))
    }

    it("ignores feature counts when determining significance") {
      val result = multinomialmodel.getSignificantDimensions(vectors(9)._2, 0.5f)
      result.size shouldBe 3
      result(0) shouldBe (0, Vectors.sparse(46, Array(), Array()))
      result(1) shouldBe (1, Vectors.sparse(46, Array(), Array()))
      result(2) shouldBe (2, Vectors.sparse(46, Array(45), Array(0.6097225849737393)))
    }
  }

  describe("gets features used tests") {
    it("Gets features used in binary case where all are used") {
      val model = new IdibonSparkMLLIBLRWrapper(
        Vectors.sparse(10,
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
          Array(0.6, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)).toDense, 0.0, 10, 2)
      model.getFeaturesUsed() shouldBe Vectors.sparse(10,
        Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        Array(0.6, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
    }
    it("Gets features used in binary case where only some are used") {
      val model = new IdibonSparkMLLIBLRWrapper(
        Vectors.sparse(10, Array(0, 3, 5), Array(0.6, 0.8, 0.1)).toDense, 0.0, 10, 2)
      model.getFeaturesUsed() shouldBe Vectors.sparse(10, Array(0, 3, 5), Array(0.6, 0.8, 0.1))
    }
    it("Gets features used in n-ary case where all are used") {
      val multinomialmodel = new IdibonSparkMLLIBLRWrapper(
        Vectors.sparse(20,
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
          Array(0.6, 0.8, 0.1, 0.3, 0.4, 0.05, 0.6, 0.8, 0.1, 0.3, 0.4, 0.05, 0.6, 0.8, 0.1, 0.3, 0.4, 0.05, 0.1, 0.1)
        ).toDense, 0.0, 10, 3)
      multinomialmodel.getFeaturesUsed() shouldBe Vectors.sparse(10,
        Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        Array(0.6, 0.8, 0.1, 0.3, 0.4, 0.05, 0.6, 0.8, 0.1, 0.3)
      )
    }
    it("Gets features used in n-ary case where only some are used") {
      val multinomialmodel = new IdibonSparkMLLIBLRWrapper(
        Vectors.sparse(20, Array(0, 3, 5, 10, 13, 15), Array(0.6, 0.8, 0.1, 0.3, 0.4, 0.05)).toDense, 0.0, 10, 3)
      multinomialmodel.getFeaturesUsed() shouldBe Vectors.sparse(10,
        Array(0, 3, 5), Array(0.6, 0.8, 0.1))
    }

  }
}
