FewRel provides the `h` (head) and `t` (tail), corresponding for the two entities involved in the relation. The format is that each one is a list, containing `3` elements:
- element 1 is the entity (a string, even if the entity consists of multiple tokens; when there are multiple tokens the string is probably constructed using ' '.join(<..>))
- element 2 is the id of the entity (FewRel used WikiData)
- element 3 is a list containing a single element:
    - element 1 is a list containing the indices of the tokens involved in the entity

Paper can be found at [FewRel1.0](https://aclanthology.org/D18-1514.pdf) and [FewRel2.0](https://aclanthology.org/D19-1649.pdf)