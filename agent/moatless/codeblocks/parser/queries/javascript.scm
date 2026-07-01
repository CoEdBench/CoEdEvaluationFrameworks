(program . (_) @child.first @definition.module) @root

(class_declaration
  name: (identifier) @identifier
  body: (class_body
    ("{") @child.first
  )
) @root @definition.class

(function_declaration
  name: (identifier) @identifier
  body: (statement_block
    ("{") @child.first
  )
) @root @definition.function

(method_definition
  name: (property_identifier) @identifier
  body: (statement_block
    ("{") @child.first
  )
) @root @definition.function

(import_statement
  (string
    (string_fragment) @reference.identifier @identifier
  )
) @root @definition.import @reference.imports

(call_expression
  function: [
    (identifier) @reference.identifier
    (member_expression) @reference.identifier
  ]
) @root @definition.call

(comment) @root @definition.comment

(_
  (statement_block
    . ("{") @child.first
  )
) @root @definition.statement
