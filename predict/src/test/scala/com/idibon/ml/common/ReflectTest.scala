package com.idibon.ml.common

import scala.reflect.runtime.universe.typeOf
import org.scalatest.{Matchers, FunSpec}
import com.idibon.ml.common.Reflect._

class ReflectSpec extends FunSpec with Matchers {

  describe("getMethodsNamed") {

    it("should return None for invalid methods") {
      getMethodsNamed("hello", "NotAMethod") shouldBe None
    }

    it("should return all overloaded variations") {
      // 4 variations of this method, all return an integer
      getMethodsNamed("hello", "indexOf")
        .map(_.map(_.returnType)) shouldBe
      Some(List(typeOf[Int], typeOf[Int], typeOf[Int], typeOf[Int]))
    }
  }

  describe("getVariadicParameterType") {
    it("should return type arguments for variadic methods") {
      val cases = List(
        "actuallyVariadic" -> Some(typeOf[Int]),
        "possiblyVariadic" -> Some(typeOf[Int]),
        "neverVariadic" -> None
      )
      val obj = new TestGetVariadicParamType
      cases.foreach({ case (methodName, expected) => {
        val method = getMethodsNamed(obj, methodName).get.head
        val variadicType = getVariadicParameterType(method.paramLists.head)
        getVariadicParameterType(method.paramLists.head) shouldBe expected
      }})
    }
  }

  describe("isValidInvocation") {
    val invoker = new TestIsValidInvocation

    it("should return true iff correct parameters are passed") {
      val simpleTests = List(
        List("m0", "m1", "m2") -> true,
        List("m0", "m1") -> false,
        List("m0", "m2", "m1") -> false,
        List("m0", "m1", "m2", "m2") -> false
      )

      val simple = getMethodsNamed(invoker, "simple").get.head

      simpleTests.foreach({ case (inputArgs, expected) => {
        val inputs = inputArgs.map(getMethodsNamed(invoker, _)
          .get.head.returnType)

        isValidInvocation(simple, List(inputs)) shouldBe expected
      }})
    }

    it("should support variadic parameters") {
      val variadicTests = List(
        List("m0", "m2") -> true,
        List("m0", "m2", "m2") -> true,
        List("m0") -> true,
        List("m0", "m3") -> true,
        List("m0", "m2", "m2", "m2") -> true,
        List("m0", "m1") -> false,
        List("m0", "m3", "m3") -> false,
        List("m0", "m2", "m2", "m1") -> false,
        List("m0", "m3", "m1") -> false
      )

      val variadic = getMethodsNamed(invoker, "variadic").get.head

      variadicTests.foreach({ case (inputArgs, expected) => {
        val inputs = inputArgs.map(getMethodsNamed(invoker, _)
          .get.head.returnType)

        isValidInvocation(variadic, List(inputs)) shouldBe expected
      }})
    }

    it("should support method currying") {
      val curryTests = List(
        List(List("m0"), List("m1")) -> true,
        List(List("m0")) -> false,
        List(List("m0"), List("m2")) -> false,
        List(List("m0"), List("m1"), List("m0")) -> false,
        List(List("m0", "m1"), List("m1")) -> false
      )

      val curried = getMethodsNamed(invoker, "curried").get.head

      curryTests.foreach({ case (inputArgs, expected) => {
        val inputs = inputArgs.map(_.map(getMethodsNamed(invoker, _)
          .get.head.returnType))

        isValidInvocation(curried, inputs) shouldBe expected
      }})
    }
  }
}

private[this] class TestGetVariadicParamType {
  def actuallyVariadic(x: Int*) = Unit
  def possiblyVariadic(x: Seq[Int]) = Unit
  def neverVariadic(x: Int) = Unit
}

private[this] class TestIsValidInvocation {
  def m0: Int = 0
  def m1: Int = 1
  def m2: String = "hi"
  def m3: Seq[String] = List("hello", "world")
  def simple(a: Int, b: Int, c: String) = Unit
  def variadic(a: Int, c: String*) = Unit
  def curried(a: Int)(b: Int) = Unit
}
