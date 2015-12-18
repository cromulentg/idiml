import scala.reflect.runtime.universe._

package com.idibon.ml.common {

  /** Variety of helper functions for handling erased reflective types
    *
    * Refer to the unit tests for usage examples.
    */
  package object Reflect {

    /** Returns all overloaded variations of method "name" in an object
      *
      * @param obj  object to reflect
      * @param name name of method to lookup
      * @return a list of methods if one or more methods is found, or None
      */
    def getMethodsNamed(obj: Any, name: String): Option[List[MethodSymbol]] = {
      Mirror.reflect(obj).symbol.typeSignature.member(TermName(name)) match {
        case term: TermSymbol if term.alternatives.exists(_.isMethod) =>
          Some(term.alternatives.filter(_.isMethod).map(_.asMethod))
        case _ => None
      }
    }

    /** Returns a method mirror for invoking the named method on obj
      *
      * Returns None if no method exists with the requested name, or if
      * multiple overloaded methods with the same name exist.
      *
      * @param obj  object to reflect
      * @param name name of method to lookup
      * @return a mirror to invoke the named method on the object
      */
    def getMethod(obj: Any, name: String): Option[MethodMirror] = {
      val instanceMirror = Mirror.reflect(obj)
      instanceMirror.symbol.typeSignature.member(TermName(name)) match {
        case meth: MethodSymbol => Some(instanceMirror.reflectMethod(meth))
        case _ => None
      }
    }

    /** Returns the type of the variadic argument accepted by the param list
      *
      * Parameter lists are treated as variadic if the last parameter has
      * a true, variadic signature (indicated by its symbol being equal to
      * definitions.RepeatedParamClass), or its erasure is exactly a Seq[Any],
      * which seems to be how a number of classes (including anonymous
      * classes) are generated.
      *
      * Returns the type argument for the variadic (i.e., the parameter T
      * within Seq[T] or T*) if the parameter list is variadic, or None for
      * non-variadic methods.
      *
      * NB: anonymous classes suffer badly from erasure (at least in
      * Scala 2.11), so the best you'll get is a Some(Object) if you call
      * getVariadicParameterType on an instance of an anonymous class
      *
      * @param params a parameter list
      * @return the type of the variadic argument accepted, or None
      */
    def getVariadicParameterType(params: List[Symbol]): Option[Type] = {
      if (params.isEmpty) {
        None
      } else {
        /* extract the type of the last (and only possible variadic) entry
         * from the parameter list */
        val lastParam = params.last.asTerm.typeSignature

        if (lastParam.typeSymbol == definitions.RepeatedParamClass)
          Some(lastParam.typeArgs.head)
        else if (lastParam.erasure =:= typeOf[Seq[Any]])
          Some(lastParam.typeArgs.head)
        else
          None
      }
    }

    /** Returns true if the curry arguments meet method's type signature
      *
      * @param method method that will be called
      * @param currys types of the arguments that will be provided on invocation
      *   to each of method's curried parameter lists
      * @return true if method's parameters will be satisified by currys
      */
    def isValidInvocation(method: Symbol, currys: List[List[Type]]): Boolean = {
      val paramLists = method.asMethod.paramLists

      /* checks the arguments supplied to a single call will satisfy
       * the parameters in the provided parameter list. */
      def validList(params: List[Symbol], args: List[Type]): Boolean = {

        val variadicType = getVariadicParameterType(params)

        /* we need to detect if the arguments treat the variadic parameter
         * as a (splatted) variadic invocation, or collate the arguments
         * into a List already. in the latter case, the number of arguments
         * to the function must match the number of parameters */
        var variadicDetected: Option[Boolean] = variadicType match {
          // auto-detect variadic invocations at the first NoSymbol entry
          case Some(_) => None
          // no variadic parameter, so all arguments must match their types
          case None => Some(false)
        }

        /* if the method is variadic, turn the last parameter into
         * a NoSymbol placeholder and add a NoSymbol placeholder for each
         * extra argument that should be checked with the variadic. if
         * fewer arguments are supplied than parameters in the signature,
         * add a NoType placeholder, then evaluate each (param, type) pair
         * for type conformance */
        variadicType.map(_ => params.dropRight(1)).getOrElse(params)
          .zipAll(args, NoSymbol, NoType)
          .forall({ case (param, arg) => {
            param match {
              // argument not provided. parameter must have a default or implicit
              case term: TermSymbol if arg == NoType =>
                term.isParamWithDefault || term.isImplicit
              // argument must conform to term's type signature
              case term: TermSymbol =>
                arg <:< term.typeSignature
              // handle the variadic argument
              case NoSymbol => variadicDetected match {
                /* invocation is variadic, all arguments must conform to the
                 * inner type (i.e., T in Seq[T]) */
                case Some(true) => arg <:< variadicType.get
                // not variadic, extra parameters are disallowed
                case Some(false) => false
                /* auto-detected as the first entry in a variadic invocation,
                 * so all subsequent arguments must be variadic */
                case None if arg <:< variadicType.get => {
                  variadicDetected = Some(true)
                  true
                }
                /* non-variadic invocation. no more arguments should exist,
                 * and the current argument should be a Seq with a
                 * parameter type conforming to the variadic type */
                case None if variadicType.isDefined => {
                  variadicDetected = Some(false)
                  (arg.typeSymbol.asType.toType <:< typeOf[Seq[_]] &&
                    arg.typeArgs.head <:< variadicType.get)
                }
              }
              case _ => false
            }
          }})
      }

      // verify that the provided argument lists match all expected parameters
      method.asMethod.paramLists
        .zipAll(currys, List(), List())
        .forall({ case (paramList, argList) => validList(paramList, argList) })
    }

    // cached mirror used for all reflection needs
    private val Mirror = runtimeMirror(getClass.getClassLoader)
  }
}
