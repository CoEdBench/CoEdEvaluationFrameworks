(source_file . (_) @child.first @definition.module) @root

(type_declaration
  (type_spec
    name: (type_identifier) @identifier
    type: [
      (struct_type
        (field_declaration_list
          ("{") @child.first
        )
      )
      (_)
    ]
  )
) @root @definition.class

(function_declaration
  name: (identifier) @identifier
  body: (block
    ("{") @child.first
  )
) @root @definition.function

(method_declaration
  name: (field_identifier) @identifier
  body: (block
    ("{") @child.first
  )
) @root @definition.function

(import_spec
  path: (interpreted_string_literal) @reference.identifier @identifier
) @root @definition.import @reference.imports

(call_expression
  function: [
    (identifier) @reference.identifier
    (selector_expression) @reference.identifier
  ]
) @root @definition.call

(comment) @root @definition.comment

(_
  (block
    . ("{") @child.first
  )
) @root @definition.statement
