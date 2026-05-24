```markdown
# pysad Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the `pysad` TypeScript repository. You'll learn how to structure files, write imports and exports, follow commit message conventions, and run or write tests. This guide is designed to help new contributors quickly get up to speed with the project's style and workflows.

## Coding Conventions

### File Naming
- Use **camelCase** for file names.
  - Example: `myModule.ts`, `dataProcessor.ts`

### Import Style
- Use **relative imports** for internal modules.
  - Example:
    ```typescript
    import { processData } from './dataProcessor';
    ```

### Export Style
- Use **named exports** for all modules.
  - Example:
    ```typescript
    // dataProcessor.ts
    export function processData(input: string): string {
      // ...
    }
    ```

### Commit Messages
- Follow **Conventional Commits**.
- Use prefixes like `ci` for continuous integration-related changes.
- Keep commit messages concise (average ~29 characters).
  - Example:
    ```
    ci: update build pipeline config
    ```

## Workflows

### Committing Changes
**Trigger:** When making any code or configuration changes  
**Command:** `/commit-changes`

1. Make your code or configuration changes.
2. Stage your changes:  
   ```
   git add .
   ```
3. Write a conventional commit message, using a prefix like `ci` if appropriate:  
   ```
   git commit -m "ci: update build pipeline config"
   ```
4. Push your changes:  
   ```
   git push
   ```

### Writing and Running Tests
**Trigger:** When adding new features or fixing bugs  
**Command:** `/run-tests`

1. Create or update test files using the `*.test.*` pattern (e.g., `myModule.test.ts`).
2. Write tests for your code (testing framework is not specified; check existing tests for patterns).
3. Run the test suite (command depends on the test framework; try `npm test` or `yarn test`).

   Example test file:
   ```typescript
   // myModule.test.ts
   import { myFunction } from './myModule';

   describe('myFunction', () => {
     it('should return true for valid input', () => {
       expect(myFunction('valid')).toBe(true);
     });
   });
   ```

## Testing Patterns

- Test files follow the `*.test.*` naming convention (e.g., `feature.test.ts`).
- The testing framework is not specified; review existing test files for patterns.
- Place tests alongside the modules they test or in a dedicated `tests` directory if present.

## Commands
| Command         | Purpose                                      |
|-----------------|----------------------------------------------|
| /commit-changes | Guide for making and committing changes      |
| /run-tests      | Steps to write and execute tests             |
```