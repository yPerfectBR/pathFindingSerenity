import { Entity, Dimension } from "@serenityjs/core";
import { BlockPosition, Vector3f, Rotation } from "@serenityjs/protocol";

// ============================================================================
// PERFORMANCE AND LIMIT CONFIGURATIONS
// ============================================================================

/**
 * Navigation system update interval in milliseconds.
 * Lower values = more frequent updates (more precise, more CPU).
 * Higher values = less frequent updates (less precise, less CPU).
 * Default: 50ms = 20 updates per second (20 TPS)
 */
const TICK_INTERVAL = 50;

/**
 * Maximum number of A* algorithm iterations before giving up.
 * Higher values = allows more complex paths (more CPU, more time).
 * Lower values = fails faster on complex paths (less CPU, faster).
 * Default: 2000 iterations
 */
const MAX_ITERATIONS = 2000;

/**
 * Maximum distance in blocks that an entity can navigate.
 * Higher values = allows longer paths (more CPU, more memory).
 * Lower values = limits to shorter paths (less CPU, less memory).
 * Default: 100 blocks
 */
const MAX_PATH_DISTANCE = 100;

/**
 * Maximum search radius when destination is blocked (in blocks).
 * Higher values = searches farther for alternative destination (more CPU).
 * Lower values = searches closer (less CPU, may fail if no nearby position found).
 * Default: 5 blocks radius
 */
const MAX_ALTERNATIVE_SEARCH_RADIUS = 5;

/**
 * Maximum time in milliseconds to calculate a path before timeout.
 * Higher values = allows longer calculations (may freeze server).
 * Lower values = cancels slow calculations faster (better responsiveness).
 * Default: 100ms
 */
const PATHFINDING_TIMEOUT = 100;

/**
 * Maximum number of simultaneous pathfinding calculations.
 * Higher values = more NPCs can calculate paths at the same time (more CPU).
 * Lower values = fewer simultaneous calculations (less overhead, more queue).
 * Default: 3 simultaneous calculations
 */
const MAX_CONCURRENT_PATHFINDING = 3;

/**
 * Block cache time-to-live in milliseconds.
 * Higher values = cache lasts longer (fewer getBlock calls, data may be stale).
 * Lower values = cache expires faster (more getBlock calls, more up-to-date data).
 * Default: 1000ms (1 second)
 */
const BLOCK_CACHE_TTL = 1000;

/**
 * Minimum distance in blocks to consider that the entity has reached the point.
 * Lower values = needs to get closer (more precise, more adjustments).
 * Higher values = accepts being farther away (less precise, fewer adjustments).
 * Default: 0.5 blocks
 */
const ARRIVAL_DISTANCE = 0.5;

/**
 * Default navigation speed for entities (in blocks per tick).
 * Higher values = entities move faster (faster, less smooth).
 * Lower values = entities move slower (slower, smoother).
 * Default: 0.2 blocks per tick
 */
const DEFAULT_NAVIGATION_SPEED = 0.2;

/**
 * Minimum horizontal distance to calculate rotation (avoids division by zero).
 * Lower values = allows more precise rotations.
 * Higher values = ignores rotations at very small distances.
 * Default: 0.01 blocks
 */
const MIN_ROTATION_DISTANCE = 0.01;

/**
 * Minimum vertical distance to calculate pitch (avoids unnecessary calculations).
 * Lower values = allows more precise pitch adjustments.
 * Higher values = ignores small vertical adjustments.
 * Default: 0.01 blocks
 */
const MIN_PITCH_DISTANCE = 0.01;

/**
 * Jump impulse force for 1 block height jump.
 * Higher values = jumps higher.
 * Lower values = jumps lower.
 * Default: 0.5 (slightly higher to ensure jump works even with low speeds)
 */
const JUMP_IMPULSE_FORCE = 0.5;

/**
 * Minimum time in milliseconds between jump impulses.
 * Prevents applying multiple impulses in the same jump.
 * Default: 200ms (reduced to allow faster response)
 */
const JUMP_IMPULSE_COOLDOWN = 200;

/**
 * Minimum vertical movement to consider as jumping (for impulse application).
 * Lower values = more sensitive to vertical movement (more jumps detected).
 * Higher values = less sensitive (only significant jumps detected).
 * Default: 0.3 blocks
 */
const JUMP_DETECTION_THRESHOLD = 0.5;

// ============================================================================
// INTERFACES AND CLASSES
// ============================================================================

/**
 * Node used in A* algorithm
 */
interface PathNode {
  position: BlockPosition;
  adjustedY?: number; // Adjusted Y position when jumping over obstacles
  g: number; // Cost from start
  h: number; // Heuristic (estimated distance to goal)
  f: number; // f = g + h
  parent: PathNode | null;
}

/**
 * Navigation state for an entity
 */
interface NavigationState {
  path: Vector3f[];
  currentIndex: number;
  speed: number;
  lastUpdate: number;
  lastJumpImpulse: number; // Timestamp of last jump impulse applied
}

/**
 * Pathfinding request queue
 */
interface PathfindingRequest {
  entity: Entity;
  destination: Vector3f;
  resolve: (path: Vector3f[] | null) => void;
  reject: (error: Error) => void;
}

/**
 * MinHeap implementation for A* openSet
 */
class MinHeap {
  private heap: (PathNode | undefined)[] = [];

  public push(node: PathNode): void {
    this.heap.push(node);
    this.bubbleUp(this.heap.length - 1);
  }

  public pop(): PathNode | undefined {
    if (this.heap.length === 0) return undefined;
    if (this.heap.length === 1) {
      const node = this.heap[0];
      this.heap.pop();
      return node;
    }

    const min = this.heap[0];
    const last = this.heap.pop();
    if (last) {
      this.heap[0] = last;
      this.bubbleDown(0);
    }
    return min;
  }

  public isEmpty(): boolean {
    return this.heap.length === 0;
  }

  public size(): number {
    return this.heap.length;
  }

  private bubbleUp(index: number): void {
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      const parentNode = this.heap[parent];
      const currentNode = this.heap[index];
      if (!parentNode || !currentNode) break;
      if (parentNode.f <= currentNode.f) break;
      
      // Swap nodes
      const temp: PathNode | undefined = this.heap[parent];
      this.heap[parent] = this.heap[index];
      this.heap[index] = temp;
      index = parent;
    }
  }

  private bubbleDown(index: number): void {
    while (true) {
      let smallest = index;
      const left = 2 * index + 1;
      const right = 2 * index + 2;

      const smallestNode = this.heap[smallest];
      if (!smallestNode) break;

      if (left < this.heap.length) {
        const leftNode = this.heap[left];
        if (leftNode && leftNode.f < smallestNode.f) {
          smallest = left;
        }
      }
      if (right < this.heap.length) {
        const rightNode = this.heap[right];
        const smallestNode2 = this.heap[smallest];
        if (rightNode && smallestNode2 && rightNode.f < smallestNode2.f) {
          smallest = right;
        }
      }
      if (smallest === index) break;
      
      const indexNode = this.heap[index];
      const smallestNode3 = this.heap[smallest];
      if (!indexNode || !smallestNode3) break;
      
      // Swap nodes
      const temp: PathNode | undefined = this.heap[index];
      this.heap[index] = this.heap[smallest];
      this.heap[smallest] = temp;
      index = smallest;
    }
  }
}

/**
 * Block cache for pathfinding
 */
class BlockCache {
  private cache = new Map<string, { isSolid: boolean; timestamp: number }>();

  public get(dimension: Dimension, position: BlockPosition): boolean {
    const key = `${position.x},${position.y},${position.z}`;
    const cached = this.cache.get(key);
    const now = Date.now();

    if (cached && now - cached.timestamp < BLOCK_CACHE_TTL) {
      return cached.isSolid;
    }

    const block = dimension.getBlock(position);
    const isSolid = block.isSolid;
    this.cache.set(key, { isSolid, timestamp: now });
    return isSolid;
  }

  public clear(): void {
    this.cache.clear();
  }
}

/**
 * Map of active navigations
 */
const activeNavigations = new Map<Entity, NavigationState>();

/**
 * Pathfinding request queue
 */
const pathfindingQueue: PathfindingRequest[] = [];
let activePathfindingCount = 0;

/**
 * Global tick interval for navigation updates
 */
let globalTickInterval: NodeJS.Timeout | null = null;

/**
 * Block cache instance
 */
const blockCache = new BlockCache();

/**
 * A* pathfinding navigation system
 */
export class NavigationSystem {
  /**
   * Initializes the navigation system
   */
  public static initialize(): void {
    if (globalTickInterval) return;

    globalTickInterval = setInterval(() => {
      this.processNavigationTick();
      this.processPathfindingQueue();
    }, TICK_INTERVAL);
  }

  /**
   * Shuts down the navigation system
   */
  public static shutdown(): void {
    if (globalTickInterval) {
      clearInterval(globalTickInterval);
      globalTickInterval = null;
    }
    activeNavigations.clear();
    pathfindingQueue.length = 0;
    blockCache.clear();
  }

  /**
   * Processes navigation updates for all entities
   */
  private static processNavigationTick(): void {
    const now = Date.now();
    const entitiesToRemove: Entity[] = [];
    const arrivalDistanceSquared = ARRIVAL_DISTANCE * ARRIVAL_DISTANCE;

    for (const [entity, state] of activeNavigations.entries()) {
      // Quick validation checks
      if (!entity.isAlive || !entity.dimension || state.currentIndex >= state.path.length) {
        entitiesToRemove.push(entity);
        continue;
      }

      const target = state.path[state.currentIndex];
      if (!target) {
        entitiesToRemove.push(entity);
        continue;
      }

      // Throttle updates
      if (now - state.lastUpdate < TICK_INTERVAL) {
        continue;
      }
      state.lastUpdate = now;

      const currentPos = entity.position;
      const dx = target.x - currentPos.x;
      const dy = target.y - currentPos.y;
      const dz = target.z - currentPos.z;
      const distanceSquared = dx * dx + dy * dy + dz * dz;

      if (distanceSquared < arrivalDistanceSquared) {
        state.currentIndex++;
        if (state.currentIndex >= state.path.length) {
          entitiesToRemove.push(entity);
          continue;
        }
      } else {
        const distance = Math.sqrt(distanceSquared);
        const normalizedDistance = Math.max(distance, 0.01);
        
        // Check movement type
        const isJumping = dy > JUMP_DETECTION_THRESHOLD;
        const isDescending = dy < -0.2;
        const isHorizontal = Math.abs(dy) <= 0.2;
        const timeSinceLastJump = now - state.lastJumpImpulse;
        
        // Apply jump impulse if needed
        let justAppliedJump = false;
        if (isJumping && timeSinceLastJump > JUMP_IMPULSE_COOLDOWN) {
          // More strict ground check - entity must be on ground or very close to it
          let isOnGround = false;
          
          if (entity.onGround !== undefined) {
            isOnGround = entity.onGround;
          } else {
            // Fallback: check if velocity.y is very low or negative (near ground or just landed)
            // Also check if entity is not moving upward significantly
            isOnGround = entity.velocity.y <= 0.05 && entity.velocity.y >= -0.1;
          }
          
          if (isOnGround) {
            entity.applyImpulse(new Vector3f(0, JUMP_IMPULSE_FORCE, 0));
            state.lastJumpImpulse = now;
            justAppliedJump = true;
          }
        }
        
        // Apply horizontal movement
        const horizontalDistanceSquared = dx * dx + dz * dz;
        if (horizontalDistanceSquared > 0.0001) {
          const horizontalDistance = Math.sqrt(horizontalDistanceSquared);
          const invHorizontal = 1 / Math.max(horizontalDistance, 0.01);
          entity.velocity.x = dx * invHorizontal * state.speed;
          entity.velocity.z = dz * invHorizontal * state.speed;
        } else {
          entity.velocity.x = 0;
          entity.velocity.z = 0;
        }
        
        // Handle vertical movement - only set if necessary
        // CRITICAL: Don't override velocity.y immediately after applying impulse
        if (!isHorizontal) {
          if (isDescending && entity.velocity.y > -0.3) {
            entity.velocity.y = Math.min((dy / normalizedDistance) * state.speed, -0.1);
          } else if (isJumping) {
            // When jumping, don't set velocity.y immediately after impulse
            // Only set if impulse was applied a long time ago and we're still way below target
            // This prevents overriding the impulse
            if (timeSinceLastJump > 300 && !justAppliedJump && dy > 0.5) {
              // Only if we're significantly below target and jump impulse seems insufficient
              entity.velocity.y = (dy / normalizedDistance) * state.speed;
            }
            // Otherwise, let the impulse work - don't touch velocity.y
          } else if (Math.abs(dy) > 0.4) {
            entity.velocity.y = (dy / normalizedDistance) * state.speed;
          }
        }
        
        this.updateEntityRotation(entity, target);
      }
    }

    // Remove completed or invalid navigations
    for (const entity of entitiesToRemove) {
      this.stopNavigation(entity);
    }
  }

  /**
   * Processes pathfinding queue
   */
  private static processPathfindingQueue(): void {
    while (
      pathfindingQueue.length > 0 &&
      activePathfindingCount < MAX_CONCURRENT_PATHFINDING
    ) {
      const request = pathfindingQueue.shift();
      if (!request) break;

      activePathfindingCount++;
      this.findPathAsync(request)
        .then((path) => {
          request.resolve(path);
        })
        .catch((error) => {
          request.reject(error);
        })
        .finally(() => {
          activePathfindingCount--;
        });
    }
  }

  /**
   * Navigates entity to destination automatically
   */
  public static navigateTo(
    entity: Entity,
    destination: Vector3f,
    speed: number = DEFAULT_NAVIGATION_SPEED
  ): boolean {
    this.stopNavigation(entity);

    // Initialize system if not already done
    this.initialize();

    // Check distance limit (Manhattan distance)
    const startPos = BlockPosition.fromVector3f(entity.position);
    const endPos = BlockPosition.fromVector3f(destination);
    const dx = Math.abs(startPos.x - endPos.x);
    const dy = Math.abs(startPos.y - endPos.y);
    const dz = Math.abs(startPos.z - endPos.z);
    if (dx + dy + dz > MAX_PATH_DISTANCE) {
      return false;
    }

    // Try synchronous pathfinding first (for immediate response)
    const path = this.findPath(entity, destination);
    if (!path || path.length === 0) {
      return false;
    }

    const state: NavigationState = {
      path,
      currentIndex: 0,
      speed,
      lastUpdate: Date.now(),
      lastJumpImpulse: 0
    };

    activeNavigations.set(entity, state);
    return true;
  }

  /**
   * Navigates entity to destination asynchronously (for long paths)
   */
  public static async navigateToAsync(
    entity: Entity,
    destination: Vector3f,
    speed: number = DEFAULT_NAVIGATION_SPEED
  ): Promise<boolean> {
    this.stopNavigation(entity);
    this.initialize();

    return new Promise((resolve, reject) => {
      pathfindingQueue.push({
        entity,
        destination,
        resolve: (path) => {
          if (!path || path.length === 0) {
            resolve(false);
            return;
          }

          const state: NavigationState = {
            path,
            currentIndex: 0,
            speed,
            lastUpdate: Date.now(),
            lastJumpImpulse: 0
          };

          activeNavigations.set(entity, state);
          resolve(true);
        },
        reject
      });
    });
  }

  /**
   * Stops navigation for an entity
   */
  public static stopNavigation(entity: Entity): void {
    const state = activeNavigations.get(entity);
    if (state) {
      activeNavigations.delete(entity);
      entity.velocity.x = 0;
      // Don't reset velocity.y - let gravity handle it naturally
      // entity.velocity.y = 0;
      entity.velocity.z = 0;
    }
  }

  /**
   * Checks if entity is navigating
   */
  public static isNavigating(entity: Entity): boolean {
    return activeNavigations.has(entity);
  }

  /**
   * Finds path from start to destination using A* (async version)
   */
  private static async findPathAsync(
    request: PathfindingRequest
  ): Promise<Vector3f[] | null> {
    return new Promise((resolve) => {
      setTimeout(() => {
        const path = this.findPath(request.entity, request.destination);
        resolve(path);
      }, 0);
    });
  }

  /**
   * Checks if a position is walkable (entity can stand there)
   */
  private static isPositionWalkable(
    dimension: Dimension,
    position: BlockPosition
  ): boolean {
    const blockAt = blockCache.get(dimension, position);
    const blockAbove = blockCache.get(dimension, new BlockPosition(position.x, position.y + 1, position.z));
    const blockBelow = blockCache.get(dimension, new BlockPosition(position.x, position.y - 1, position.z));
    return !blockAt && !blockAbove && blockBelow;
  }

  /**
   * Finds the nearest accessible position to the target destination
   */
  private static findNearestAccessiblePosition(
    dimension: Dimension,
    targetPos: BlockPosition,
    startPos: BlockPosition
  ): BlockPosition | null {
    // First check if target position itself is walkable
    if (this.isPositionWalkable(dimension, targetPos)) {
      return targetPos;
    }

    // Search in expanding radius around target
    let bestPos: BlockPosition | null = null;
    let bestDistance = Infinity;

    for (let radius = 1; radius <= MAX_ALTERNATIVE_SEARCH_RADIUS; radius++) {
      for (let dx = -radius; dx <= radius; dx++) {
        for (let dz = -radius; dz <= radius; dz++) {
          // Only check positions on the edge of current radius
          const isOnEdge = Math.abs(dx) === radius || Math.abs(dz) === radius;
          if (!isOnEdge && radius > 1) continue;

          // Check multiple Y levels (same level, one above, one below)
          for (let dy = -1; dy <= 1; dy++) {
            const candidatePos = new BlockPosition(
              targetPos.x + dx,
              targetPos.y + dy,
              targetPos.z + dz
            );

            if (this.isPositionWalkable(dimension, candidatePos)) {
              // Calculate distance from start (prefer closer to start)
              const distance = Math.abs(candidatePos.x - startPos.x) +
                Math.abs(candidatePos.y - startPos.y) +
                Math.abs(candidatePos.z - startPos.z);

              // Also consider distance from original target (prefer closer to target)
              const distanceFromTarget = Math.abs(candidatePos.x - targetPos.x) +
                Math.abs(candidatePos.y - targetPos.y) +
                Math.abs(candidatePos.z - targetPos.z);

              // Combined score: prefer positions closer to both start and target
              const score = distance + distanceFromTarget * 0.5;

              if (score < bestDistance) {
                bestDistance = score;
                bestPos = candidatePos;
              }
            }
          }
        }
      }

      // If we found a position, return it (don't search further)
      if (bestPos) {
        return bestPos;
      }
    }

    return bestPos;
  }

  /**
   * Finds path from start to destination using A*
   */
  public static findPath(
    entity: Entity,
    destination: Vector3f
  ): Vector3f[] | null {
    const dimension: Dimension | undefined = entity.dimension;
    if (!dimension) {
      return null;
    }

    const startPos = BlockPosition.fromVector3f(entity.position);
    let endPos = BlockPosition.fromVector3f(destination);

    // Check if destination is walkable, if not find nearest accessible position
    if (!this.isPositionWalkable(dimension, endPos)) {
      const alternativePos = this.findNearestAccessiblePosition(dimension, endPos, startPos);
      if (alternativePos) {
        endPos = alternativePos;
      } else {
        // No accessible position found nearby
        return null;
      }
    }

    if (this.positionsEqual(startPos, endPos)) {
      // Convert back to Vector3f for return
      return [new Vector3f(endPos.x + 0.5, endPos.y + 0.5, endPos.z + 0.5)];
    }

    // Check distance limit (Manhattan distance - faster than Euclidean)
    const dx = Math.abs(startPos.x - endPos.x);
    const dy = Math.abs(startPos.y - endPos.y);
    const dz = Math.abs(startPos.z - endPos.z);
    if (dx + dy + dz > MAX_PATH_DISTANCE) {
      return null;
    }

    const openSet = new MinHeap();
    const openSetMap = new Map<string, PathNode>(); // Fast lookup
    const closedSet = new Set<string>();

    const startNode: PathNode = {
      position: startPos,
      g: 0,
      h: this.heuristic(startPos, endPos),
      f: 0,
      parent: null
    };
    startNode.f = startNode.g + startNode.h;
    openSet.push(startNode);
    openSetMap.set(this.positionKey(startPos), startNode);

    let iterations = 0;
    const startTime = Date.now();

    while (!openSet.isEmpty() && iterations < MAX_ITERATIONS) {
      iterations++;

      // Timeout check
      if (Date.now() - startTime > PATHFINDING_TIMEOUT) {
        return null;
      }
      const current = openSet.pop();
      if (!current) break;

      const currentKey = this.positionKey(current.position);
      openSetMap.delete(currentKey);

      if (this.positionsEqual(current.position, endPos)) {
        return this.reconstructPath(current);
      }

      closedSet.add(currentKey);

      const neighbors = this.getNeighbors(current.position);
      for (const neighborPos of neighbors) {
        const neighborKey = this.positionKey(neighborPos);

        if (closedSet.has(neighborKey)) {
          continue;
        }

        // Calculate movement deltas
        const dx = neighborPos.x - current.position.x;
        const dz = neighborPos.z - current.position.z;
        const dy = neighborPos.y - current.position.y;
        const isDiagonal = dx !== 0 && dz !== 0;
        
        // Check if diagonal movement requires three-step path (straight -> up -> final)
        let requiresThreeStep = false;
        if (isDiagonal && dy === 1) {
          // Explicit diagonal jump - always use three steps
          requiresThreeStep = true;
        } else if (isDiagonal && dy === 0) {
          // Check if we need to go up to reach diagonal position
          const blockAtDest = blockCache.get(dimension, neighborPos);
          const blockBelowDest = blockCache.get(dimension, new BlockPosition(neighborPos.x, neighborPos.y - 1, neighborPos.z));
          
          if (blockAtDest) {
            // Block at destination - check if we can go up
            const posOneUp = new BlockPosition(neighborPos.x, neighborPos.y + 1, neighborPos.z);
            if (!blockCache.get(dimension, posOneUp) && !blockCache.get(dimension, new BlockPosition(posOneUp.x, posOneUp.y + 1, posOneUp.z))) {
              requiresThreeStep = true;
            }
          } else if (!blockBelowDest) {
            // No ground - check if position above has ground
            const posOneUp = new BlockPosition(neighborPos.x, neighborPos.y + 1, neighborPos.z);
            if (!blockCache.get(dimension, posOneUp) && blockCache.get(dimension, new BlockPosition(posOneUp.x, posOneUp.y - 1, posOneUp.z))) {
              requiresThreeStep = true;
            }
          }
          // Check if lateral paths are blocked
          if (!requiresThreeStep) {
            const lateralX = new BlockPosition(current.position.x + dx, current.position.y, current.position.z);
            const lateralZ = new BlockPosition(current.position.x, current.position.y, current.position.z + dz);
            if (blockCache.get(dimension, lateralX) || blockCache.get(dimension, lateralZ)) {
              requiresThreeStep = true;
            }
          }
        }
        
        // For diagonal movements that require three-step path, go straight first, then up, then to final position
        if (requiresThreeStep && isDiagonal) {
          // Helper to add three-step path: lateral -> vertical -> final
          const addThreeStepPath = (lateral: BlockPosition, finalX: number, finalZ: number): void => {
            const lateralKey = this.positionKey(lateral);
            if (closedSet.has(lateralKey)) return;
            
            const lateralMoveCost = this.getMoveCost(current.position, lateral, dimension);
            if (lateralMoveCost === Infinity) return;
            
            const lateralG = current.g + lateralMoveCost;
            let lateralNode = openSetMap.get(lateralKey);
            
            if (!lateralNode) {
              lateralNode = {
                position: lateral,
                g: lateralG,
                h: this.heuristic(lateral, endPos),
                f: 0,
                parent: current
              };
              lateralNode.f = lateralNode.g + lateralNode.h;
              openSet.push(lateralNode);
              openSetMap.set(lateralKey, lateralNode);
            } else if (lateralG >= lateralNode.g) {
              return;
            } else {
              lateralNode.g = lateralG;
              lateralNode.f = lateralNode.g + lateralNode.h;
              lateralNode.parent = current;
              openSet.push(lateralNode);
            }
            
            // Step 2: Move up vertically
            const vertical = new BlockPosition(lateral.x, lateral.y + 1, lateral.z);
            const verticalKey = this.positionKey(vertical);
            if (closedSet.has(verticalKey)) return;
            
            const verticalMoveCost = this.getMoveCost(lateral, vertical, dimension);
            if (verticalMoveCost === Infinity) return;
            
            const verticalG = lateralG + verticalMoveCost;
            let verticalNode = openSetMap.get(verticalKey);
            
            if (!verticalNode) {
              verticalNode = {
                position: vertical,
                g: verticalG,
                h: this.heuristic(vertical, endPos),
                f: 0,
                parent: lateralNode
              };
              verticalNode.f = verticalNode.g + verticalNode.h;
              openSet.push(verticalNode);
              openSetMap.set(verticalKey, verticalNode);
            } else if (verticalG >= verticalNode.g) {
              return;
            } else {
              verticalNode.g = verticalG;
              verticalNode.f = verticalNode.g + verticalNode.h;
              verticalNode.parent = lateralNode;
              openSet.push(verticalNode);
            }
            
            // Step 3: Move to final position
            const finalPos = new BlockPosition(finalX, vertical.y, finalZ);
            const finalKey = this.positionKey(finalPos);
            if (closedSet.has(finalKey)) return;
            
            const finalMoveCost = this.getMoveCost(vertical, finalPos, dimension);
            if (finalMoveCost === Infinity) return;
            
            const finalG = verticalG + finalMoveCost;
            let finalNode = openSetMap.get(finalKey);
            
            if (!finalNode) {
              finalNode = {
                position: finalPos,
                g: finalG,
                h: this.heuristic(finalPos, endPos),
                f: 0,
                parent: verticalNode
              };
              finalNode.f = finalNode.g + finalNode.h;
              openSet.push(finalNode);
              openSetMap.set(finalKey, finalNode);
            } else if (finalG < finalNode.g) {
              finalNode.g = finalG;
              finalNode.f = finalNode.g + finalNode.h;
              finalNode.parent = verticalNode;
              openSet.push(finalNode);
            }
          };
          
          // Try both paths: X first, then Z first
          addThreeStepPath(
            new BlockPosition(current.position.x + dx, current.position.y, current.position.z),
            neighborPos.x,
            neighborPos.z
          );
          addThreeStepPath(
            new BlockPosition(current.position.x, current.position.y, current.position.z + dz),
            neighborPos.x,
            neighborPos.z
          );
          
          continue;
        }

        // Now calculate moveCost for non-diagonal-jump movements
        const moveCost = this.getMoveCost(
          current.position,
          neighborPos,
          dimension
        );

        if (moveCost === Infinity) {
          continue;
        }

        // Check if we need to adjust Y for jumping over obstacle
        let adjustedY: number | undefined = undefined;
        if (neighborPos.y === current.position.y && blockCache.get(dimension, neighborPos)) {
          const blockAbove = new BlockPosition(neighborPos.x, neighborPos.y + 1, neighborPos.z);
          const blockAbove2 = new BlockPosition(neighborPos.x, neighborPos.y + 2, neighborPos.z);
          if (!blockCache.get(dimension, blockAbove) && !blockCache.get(dimension, blockAbove2)) {
            adjustedY = neighborPos.y + 1;
          }
        }

        // Check if this is a corner navigation (moveCost 1.5 indicates corner navigation)
        if (moveCost === 1.5 && isDiagonal && blockCache.get(dimension, neighborPos)) {
          // Corner navigation - go to one side first, then turn
          const lateralX = new BlockPosition(current.position.x + dx, current.position.y, current.position.z);
          const lateralZ = new BlockPosition(current.position.x, current.position.y, current.position.z + dz);
          const finalPos = neighborPos;
          
          // Helper function to check if position is walkable
          const isWalkable = (pos: BlockPosition): boolean => {
            const blockAt = blockCache.get(dimension, pos);
            const blockAbove = blockCache.get(dimension, new BlockPosition(pos.x, pos.y + 1, pos.z));
            const blockBelow = blockCache.get(dimension, new BlockPosition(pos.x, pos.y - 1, pos.z));
            return !blockAt && !blockAbove && blockBelow;
          };
          
          // Helper function to add lateral path
          const addLateralPath = (lateral: BlockPosition, final: BlockPosition): void => {
            if (!isWalkable(lateral)) return;
            
            const lateralKey = this.positionKey(lateral);
            if (closedSet.has(lateralKey)) return;
            
            const lateralMoveCost = this.getMoveCost(current.position, lateral, dimension);
            if (lateralMoveCost === Infinity) return;
            
            const lateralG = current.g + lateralMoveCost;
            let lateralNode = openSetMap.get(lateralKey);
            
            if (!lateralNode) {
              lateralNode = {
                position: lateral,
                g: lateralG,
                h: this.heuristic(lateral, endPos),
                f: 0,
                parent: current
              };
              lateralNode.f = lateralNode.g + lateralNode.h;
              openSet.push(lateralNode);
              openSetMap.set(lateralKey, lateralNode);
            } else if (lateralG < lateralNode.g) {
              lateralNode.g = lateralG;
              lateralNode.f = lateralNode.g + lateralNode.h;
              lateralNode.parent = current;
              openSet.push(lateralNode);
              return;
            } else {
              return;
            }
            
            // Add final position if reachable from lateral
            if (isWalkable(final)) {
              const finalKey = this.positionKey(final);
              if (!closedSet.has(finalKey)) {
                const finalMoveCost = this.getMoveCost(lateral, final, dimension);
                if (finalMoveCost !== Infinity) {
                  const finalG = lateralG + finalMoveCost;
                  let finalNode = openSetMap.get(finalKey);
                  
                  if (!finalNode) {
                    finalNode = {
                      position: final,
                      g: finalG,
                      h: this.heuristic(final, endPos),
                      f: 0,
                      parent: lateralNode
                    };
                    finalNode.f = finalNode.g + finalNode.h;
                    openSet.push(finalNode);
                    openSetMap.set(finalKey, finalNode);
                  } else if (finalG < finalNode.g) {
                    finalNode.g = finalG;
                    finalNode.f = finalNode.g + finalNode.h;
                    finalNode.parent = lateralNode;
                    openSet.push(finalNode);
                  }
                }
              }
            }
          };
          
          // Try both lateral paths
          addLateralPath(lateralX, finalPos);
          addLateralPath(lateralZ, finalPos);
          
          // Skip the diagonal node itself
          continue;
        }

        const tentativeG = current.g + moveCost;
        const existingNode = openSetMap.get(neighborKey);

        if (!existingNode) {
          const neighborNode: PathNode = {
            position: neighborPos,
            adjustedY,
            g: tentativeG,
            h: this.heuristic(neighborPos, endPos),
            f: 0,
            parent: current
          };
          neighborNode.f = neighborNode.g + neighborNode.h;
          openSet.push(neighborNode);
          openSetMap.set(neighborKey, neighborNode);
        } else if (tentativeG < existingNode.g) {
          existingNode.g = tentativeG;
          existingNode.f = existingNode.g + existingNode.h;
          existingNode.parent = current;
          existingNode.adjustedY = adjustedY;
          // Re-add to heap with new priority
          openSet.push(existingNode);
        }
      }
    }

    return null;
  }

  /**
   * Reconstructs path from final node
   */
  private static reconstructPath(node: PathNode): Vector3f[] {
    const path: Vector3f[] = [];
    let current: PathNode | null = node;

    while (current !== null) {
      const y = current.adjustedY !== undefined ? current.adjustedY : current.position.y;
      path.unshift(
        new Vector3f(
          current.position.x + 0.5,
          y + 0.5,
          current.position.z + 0.5
        )
      );
      current = current.parent;
    }

    return path;
  }

  /**
   * Neighbor directions (static to avoid recreation)
   */
  private static readonly NEIGHBOR_DIRECTIONS = [
    { x: 1, y: 0, z: 0 }, { x: -1, y: 0, z: 0 },
    { x: 0, y: 0, z: 1 }, { x: 0, y: 0, z: -1 },
    { x: 1, y: 0, z: 1 }, { x: 1, y: 0, z: -1 },
    { x: -1, y: 0, z: 1 }, { x: -1, y: 0, z: -1 },
    { x: 0, y: 1, z: 0 }, { x: 0, y: -1, z: 0 },
    { x: 0, y: -2, z: 0 },
    { x: 1, y: -1, z: 0 }, { x: -1, y: -1, z: 0 },
    { x: 0, y: -1, z: 1 }, { x: 0, y: -1, z: -1 }
  ] as const;

  /**
   * Gets valid neighbors of a position
   */
  private static getNeighbors(position: BlockPosition): BlockPosition[] {
    const neighbors: BlockPosition[] = [];
    for (const dir of this.NEIGHBOR_DIRECTIONS) {
      neighbors.push(
        new BlockPosition(
          position.x + dir.x,
          position.y + dir.y,
          position.z + dir.z
        )
      );
    }
    return neighbors;
  }

  /**
   * Calculates movement cost between two positions
   * Returns Infinity if movement is invalid
   */
  private static getMoveCost(
    from: BlockPosition,
    to: BlockPosition,
    dimension: Dimension
  ): number {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = to.z - from.z;
    const isDiagonal = dx !== 0 && dz !== 0;

    // Vertical movement up (jump)
    if (dy === 1) {
      if (isDiagonal) return Infinity; // Force three-step path
      
      const blockAtDest = blockCache.get(dimension, to);
      const blockAboveDest = blockCache.get(dimension, new BlockPosition(to.x, to.y + 1, to.z));
      const blockBelowDest = blockCache.get(dimension, new BlockPosition(to.x, to.y - 1, to.z));

      if (!blockAtDest && !blockAboveDest && blockBelowDest) {
        return 1.5;
      }
      return Infinity;
    }

    // Vertical movement down (can descend multiple blocks)
    if (dy < 0) {
      const blockAtDest = blockCache.get(dimension, to);
      const blockAboveDest = blockCache.get(dimension, new BlockPosition(to.x, to.y + 1, to.z));
      const blockBelowDest = blockCache.get(dimension, new BlockPosition(to.x, to.y - 1, to.z));

      if (!blockAtDest && !blockAboveDest && blockBelowDest) {
        return 1.2 + (Math.abs(dy) - 1) * 0.3;
      }
      
      // Allow descending stairs (solid block at destination)
      if (blockAtDest && !blockAboveDest && !blockCache.get(dimension, new BlockPosition(to.x, to.y + 2, to.z))) {
        return 1.3 + (Math.abs(dy) - 1) * 0.3;
      }
      
      return Infinity;
    }

    // Horizontal or diagonal movement
    if (dy === 0) {
      const blockAt = blockCache.get(dimension, to);
      const blockAbove = blockCache.get(dimension, new BlockPosition(to.x, to.y + 1, to.z));
      const blockBelow = blockCache.get(dimension, new BlockPosition(to.x, to.y - 1, to.z));

      if (blockAt) {
        // Obstacle at destination
        if (isDiagonal) {
          // Check if we can go around corner
          const lateralX = new BlockPosition(from.x + dx, from.y, from.z);
          const lateralZ = new BlockPosition(from.x, from.y, from.z + dz);
          const blockLateralX = blockCache.get(dimension, lateralX);
          const blockLateralZ = blockCache.get(dimension, lateralZ);
          const blockAboveLateralX = blockCache.get(dimension, new BlockPosition(lateralX.x, lateralX.y + 1, lateralX.z));
          const blockAboveLateralZ = blockCache.get(dimension, new BlockPosition(lateralZ.x, lateralZ.y + 1, lateralZ.z));
          const blockBelowLateralX = blockCache.get(dimension, new BlockPosition(lateralX.x, lateralX.y - 1, lateralX.z));
          const blockBelowLateralZ = blockCache.get(dimension, new BlockPosition(lateralZ.x, lateralZ.y - 1, lateralZ.z));

          if ((!blockLateralX && !blockAboveLateralX && blockBelowLateralX) ||
              (!blockLateralZ && !blockAboveLateralZ && blockBelowLateralZ)) {
            return 1.5; // Corner navigation
          }
        }

        // Check if we can jump over obstacle
        if (!blockAbove && !blockCache.get(dimension, new BlockPosition(to.x, to.y + 2, to.z))) {
          return isDiagonal ? 2.0 : 1.8;
        }
        return Infinity;
      }

      // Normal movement - no obstacle
      if (blockAbove || !blockBelow) {
        return Infinity;
      }

      return isDiagonal ? 1.414 : 1.0;
    }

    return Infinity;
  }

  /**
   * Heuristic: Manhattan distance
   */
  private static heuristic(
    from: BlockPosition,
    to: BlockPosition
  ): number {
    return (
      Math.abs(from.x - to.x) +
      Math.abs(from.y - to.y) +
      Math.abs(from.z - to.z)
    );
  }

  /**
   * Compares two positions
   */
  private static positionsEqual(
    a: BlockPosition,
    b: BlockPosition
  ): boolean {
    return a.x === b.x && a.y === b.y && a.z === b.z;
  }

  /**
   * Generates unique key for position (for Set/Map)
   */
  private static positionKey(position: BlockPosition): string {
    return `${position.x},${position.y},${position.z}`;
  }

  /**
   * Updates entity rotation to look at target
   */
  private static updateEntityRotation(entity: Entity, target: Vector3f): void {
    const currentPos = entity.position;
    const dx = target.x - currentPos.x;
    const dy = target.y - currentPos.y;
    const dz = target.z - currentPos.z;
    const horizontalDistanceSquared = dx * dx + dz * dz;
    
    if (horizontalDistanceSquared < MIN_ROTATION_DISTANCE * MIN_ROTATION_DISTANCE) {
      return;
    }

    const horizontalDistance = Math.sqrt(horizontalDistanceSquared);
    let yaw = Math.atan2(dz, dx) * (180 / Math.PI) - 90;
    // Normalize yaw to [-180, 180]
    if (yaw > 180) yaw -= 360;
    else if (yaw < -180) yaw += 360;

    let pitch = 0;
    if (Math.abs(dy) > MIN_PITCH_DISTANCE) {
      pitch = -Math.atan2(dy, horizontalDistance) * (180 / Math.PI);
      if (pitch > 90) pitch = 90;
      else if (pitch < -90) pitch = -90;
    }

    entity.setRotation(new Rotation(yaw, pitch, yaw));
  }
}
