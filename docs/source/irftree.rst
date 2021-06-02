.. _irftree:

IRF Trees in CDR
=================

The network of IRFs for each CDR model are represented by the implementations in this package as *trees* from which properties can be read off for the purposes of model construction.
This is an implementation detail that will not be pertinent to most users, especially those primarily training and using CDR models through the bundled executables.
However, for users who want to program with CDR classes or directly interface with the associated ``Formula`` class, which is used to store and manipulate information about the model structure, this reference provides a conceptual overview of tree representations of CDR impulse response structures.

There are several reasons for representing IRFs as nodes on a tree. In particular, tree representations facilitate:

  - **Factorization** of separable model components, especially *impulses*, *coefficients*, and *IRFs*  (see discussion below)
  - **IRF composition**: convolving multiple IRF with each other in a hierarchical fashion
  - **Parameter tying**: sharing a single IRF or coefficient across multiple portions of the model

Note that these components cannot be cleanly factorized in CDRNN, so CDRNN "trees" are flat.
The remainder of this page applies only to CDR.


Factorization
-------------

CDR models contain *impulses*, *coefficients*, and *IRFs*, each of whose structure can be separately manipulated.
Impulses are timestamped data streams that constitute the predictors of the model.
Impulses are computed from the input data prior to entering the CDR computation graph.

Coefficients govern the amplitude of the response to a predictor.
Specifically, the IRF for a given predictor is multiplied by a coefficient governing the magnitude and sign of the overall response.
Because multiplication distributes over convolution, coefficients are only needed at the point at which an IRF is convolved with an impulse.
In hierarchical models involving convolution of multiple IRFs together, the coefficient can be seen as the product of all coefficients on all composed IRFs.
It is therefore desirable to restrict coefficients only to those IRF that "make contact" with the impulse data.

IRFs govern the shape of the response to a predictor.
The response is subsequently scaled using a coefficient.
Arbitrary IRF can be convolved together in a hierarchical fashion using a fast Fourier transform-based discrete approximation to the continuous convolution.

Tree structures permit natural separation of these three components.
Specifically, the CDR IRF tree consists of *nodes*, each of which represents the application of an IRF.
IRF composition is represented in the tree as *dominance*: parent IRF are convolved with their children, which are they convolved with their children, etc.
Coefficients are disallowed except at *terminal* IRF nodes, i.e. "leaves" of the tree that have no children.
In this implementation, ``Terminal`` is distinguished as special type of IRF kernel.
The leaves of the IRF tree must always and only be ``Terminal``.
Only ``Terminal`` IRF nodes are permitted to have coefficients.
Attempts to specify coefficients at other portions of the tree are ignored.
Terminals always have a 1-to-1 link to an impulse contained in the data.

Thus, impulses, coefficients, and IRFs are kept separate by representing the IRF as nodes in a tree and the impulses and coefficients as decorations on a distinguished ``Terminal`` node type.


IRF composition
---------------

All IRF trees are rooted at a special ``ROOT`` node, and subsequent application and composition of IRF is represented by tree structures dominated by ``ROOT``.
A simple model with no IRF composition will have a tree depth of 3::

    ROOT --> IRF --> Terminals.

Models involving composition of IRF will have greater depth, as composition of IRF is represented as dominance relations between them.
For example, imagine that the IRF for a given predictor is expected to be a Gamma convolved with another Gamma, which is then convolved with a Normal.
The path through the tree describing this structure will be::

    ROOT --> Gamma1 --> Gamma2 --> Terminal

At the terminal, the composed IRF is multiplied by its coefficient and then convolved with its associated impulse.
Although IRF composition in CDR respects the hierarchical order in which the IRF are specified, in most cases order of composition does not matter because of associativity of convolution.


Parameter tying
---------------

IRF can be tied in CDR, a property which can be conveniently represented via branching structures in the tree.
Specifically, if a single IRF is convolved with (1) multiple downstream IRF in a composition hierarchy or (2) multiple impulses, this fact can be represented by attaching multiple child nodes to a node representing the IRF.
For example, consider the following two IRF trees involving impulses A, B, and C::

  1)
              ROOT
          /     |     \
    NormalA  NormalB  NormalC
       |        |        |
     TermA    TermB    TermC

  2)
          ROOT
            |
          Normal
        /   |    \
    TermA TermB TermC

In Tree (1), each terminal has its own IRF, so the branching occurs below ``ROOT``, while in Tree (2), a single IRF is shared by all three impulses, so the branch occurs below ``IRF``.
More complex models could be formed e.g. by replacing one of the terminals in these examples with other IRF treelets, and so on.

Tying of coefficients is also supported but this is not represented in the tree structure.
It is enforced simply by requiring multiple terminals to share the same value for the coefficient ID property.


CDR tree summaries
-------------------

In the CDR model summary, the IRF tree is printed in string format with indentation representing dominance.
The ID of each IRF on the tree is printed on its own line, along with other metadata when relevant (such as trainable params and associated random grouping factors).
For example, Tree (1) above would be represented as follows::

  ROOT
    Normal.A; trainable params: mu, sigma
      Normal.A-Terminal.A
    Normal.B; trainable params: mu, sigma
      Normal.B-Terminal.B
    Normal.C; trainable params: mu, sigma
      Normal.C-Terminal.C

while Tree (2) would be represented as follows::

  ROOT
    Normal; trainable params: mu, sigma
      Normal-Terminal.A
      Normal-Terminal.B
      Normal-Terminal.C
