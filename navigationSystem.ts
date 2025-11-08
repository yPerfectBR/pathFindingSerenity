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
let isProcessingPathfinding = false;
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

    for (const [entity, state] of activeNavigations.entries()) {
      if (!entity.isAlive || !entity.dimension) {
        entitiesToRemove.push(entity);
        continue;
      }

      if (state.currentIndex >= state.path.length) {
        entitiesToRemove.push(entity);
        continue;
      }

      const target = state.path[state.currentIndex];
      if (!target) {
        entitiesToRemove.push(entity);
        continue;
      }

      // Throttle updates to avoid excessive calculations
      if (now - state.lastUpdate < TICK_INTERVAL) {
        continue;
      }
      state.lastUpdate = now;

      const currentPos = entity.position;
      const dx = target.x - currentPos.x;
      const dy = target.y - currentPos.y;
      const dz = target.z - currentPos.z;
      const distanceSquared = dx * dx + dy * dy + dz * dz;

      // Use squared distance for comparison (avoid Math.sqrt)
      const arrivalDistanceSquared = ARRIVAL_DISTANCE * ARRIVAL_DISTANCE;
      if (distanceSquared < arrivalDistanceSquared) {
        state.currentIndex++;
        if (state.currentIndex >= state.path.length) {
          entitiesToRemove.push(entity);
          continue;
        }
      } else {
        const distance = Math.sqrt(distanceSquared);
        const normalizedDistance = Math.max(distance, 0.01);
        entity.velocity.x = (dx / normalizedDistance) * state.speed;
        entity.velocity.y = (dy / normalizedDistance) * state.speed;
        entity.velocity.z = (dz / normalizedDistance) * state.speed;
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

    // Check distance limit
    const startPos = BlockPosition.fromVector3f(entity.position);
    const endPos = BlockPosition.fromVector3f(destination);
    const distance = Math.abs(startPos.x - endPos.x) +
      Math.abs(startPos.y - endPos.y) +
      Math.abs(startPos.z - endPos.z);

    if (distance > MAX_PATH_DISTANCE) {
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
      lastUpdate: Date.now()
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
            lastUpdate: Date.now()
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
      entity.velocity.y = 0;
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
    const startTime = Date.now();
    return new Promise((resolve) => {
      // Use setTimeout to allow other operations
      setTimeout(() => {
        const path = this.findPath(request.entity, request.destination);
        const elapsed = Date.now() - startTime;
        if (elapsed > PATHFINDING_TIMEOUT) {
          blockCache.clear(); // Clear cache if timeout
        }
        resolve(path);
      }, 0);
    });
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
    const endPos = BlockPosition.fromVector3f(destination);

    if (this.positionsEqual(startPos, endPos)) {
      return [destination];
    }

    // Check distance limit
    const distance = Math.abs(startPos.x - endPos.x) +
      Math.abs(startPos.y - endPos.y) +
      Math.abs(startPos.z - endPos.z);
    if (distance > MAX_PATH_DISTANCE) {
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
        if (neighborPos.y === current.position.y) {
          if (blockCache.get(dimension, neighborPos)) {
            // Obstacle of height 1, check if we can jump over
            const blockAbove = new BlockPosition(neighborPos.x, neighborPos.y + 1, neighborPos.z);
            const blockAbove2 = new BlockPosition(neighborPos.x, neighborPos.y + 2, neighborPos.z);
            if (
              !blockCache.get(dimension, blockAbove) &&
              !blockCache.get(dimension, blockAbove2)
            ) {
              adjustedY = neighborPos.y + 1;
            }
          }
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
   * Gets valid neighbors of a position
   */
  private static getNeighbors(position: BlockPosition): BlockPosition[] {
    const neighbors: BlockPosition[] = [];
    const directions = [
      { x: 1, y: 0, z: 0 },
      { x: -1, y: 0, z: 0 },
      { x: 0, y: 0, z: 1 },
      { x: 0, y: 0, z: -1 },
      { x: 1, y: 0, z: 1 },
      { x: 1, y: 0, z: -1 },
      { x: -1, y: 0, z: 1 },
      { x: -1, y: 0, z: -1 },
      { x: 0, y: 1, z: 0 },
      { x: 0, y: -1, z: 0 }
    ];

    for (const dir of directions) {
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

    // Vertical movement up (jump)
    if (dy === 1) {
      const blockAtDest = blockCache.get(dimension, to);
      const blockAboveDest = blockCache.get(
        dimension,
        new BlockPosition(to.x, to.y + 1, to.z)
      );
      const blockBelowDest = blockCache.get(
        dimension,
        new BlockPosition(to.x, to.y - 1, to.z)
      );

      if (!blockAtDest && !blockAboveDest && blockBelowDest) {
        return 1.5;
      }
      return Infinity;
    }

    // Vertical movement down
    if (dy === -1) {
      const blockAtDest = blockCache.get(dimension, to);
      const blockAboveDest = blockCache.get(
        dimension,
        new BlockPosition(to.x, to.y + 1, to.z)
      );
      const blockBelowDest = blockCache.get(
        dimension,
        new BlockPosition(to.x, to.y - 1, to.z)
      );

      if (!blockAtDest && !blockAboveDest && blockBelowDest) {
        return 1.2;
      }
      return Infinity;
    }

    // Horizontal or diagonal movement
    if (dy === 0) {
      const blockAt = blockCache.get(dimension, to);
      const blockAbove = blockCache.get(
        dimension,
        new BlockPosition(to.x, to.y + 1, to.z)
      );
      const blockBelow = blockCache.get(
        dimension,
        new BlockPosition(to.x, to.y - 1, to.z)
      );

      // If there's a solid block at destination (obstacle of height 1), check if we can jump over it
      if (blockAt) {
        // Check if we can jump over the obstacle
        const blockAboveObstacle = blockCache.get(
          dimension,
          new BlockPosition(to.x, to.y + 2, to.z)
        );

        // Can jump over if: obstacle is 1 block high, space above obstacle (y+1 and y+2) is free
        // The obstacle itself (blockAt) serves as the solid ground to jump from
        if (!blockAbove && !blockAboveObstacle) {
          // Cost for jumping over obstacle
          const isDiagonal = dx !== 0 && dz !== 0;
          return isDiagonal ? 2.0 : 1.8;
        }
        return Infinity;
      }

      // Normal horizontal movement - no obstacle
      if (blockAbove || !blockBelow) {
        return Infinity;
      }

      const isDiagonal = dx !== 0 && dz !== 0;
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
    const minRotationDistanceSquared = MIN_ROTATION_DISTANCE * MIN_ROTATION_DISTANCE;

    if (horizontalDistanceSquared < minRotationDistanceSquared) {
      return;
    }

    const horizontalDistance = Math.sqrt(horizontalDistanceSquared);
    let yaw = Math.atan2(dz, dx) * (180 / Math.PI) - 90;
    while (yaw > 180) yaw -= 360;
    while (yaw < -180) yaw += 360;

    let pitch = 0;
    if (Math.abs(dy) > MIN_PITCH_DISTANCE) {
      pitch = -Math.atan2(dy, horizontalDistance) * (180 / Math.PI);
      if (pitch > 90) pitch = 90;
      if (pitch < -90) pitch = -90;
    }

    entity.setRotation(new Rotation(yaw, pitch, yaw));
  }
}
