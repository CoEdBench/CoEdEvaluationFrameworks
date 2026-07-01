(translation_unit . (_) @child.first @definition.module) @root

(function_definition
  (function_declarator
    declarator: (_) @identifier
  )
  body: (compound_statement
    ("{") @child.first
  )
) @root @definition.function

(struct_specifier
  name: (type_identifier) @identifier
  body: (field_declaration_list
    ("{") @child.first
  )
) @root @definition.class

(enum_specifier
  name: (type_identifier) @identifier
) @root @definition.class

(preproc_include
  path: (_) @reference.identifier @identifier
) @root @definition.import @reference.imports

(call_expression
  function: [
    (identifier) @reference.identifier
    (field_expression) @reference.identifier
  ]
) @root @definition.call

(comment) @root @definition.comment

(_
  (compound_statement
    . ("{") @child.first
  )
) @root @definition.statement
