(source_file . (_) @child.first @definition.module) @root

(function_item
  name: (identifier) @identifier
  body: (block
    ("{") @child.first
  )
) @root @definition.function

(call_expression
  function: [
    (identifier) @reference.identifier
    (field_expression) @reference.identifier
  ]
) @root @definition.call

(_
  (block
    . ("{") @child.first
  )
) @root @definition.statement
