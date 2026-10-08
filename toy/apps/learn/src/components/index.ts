import { Board } from "./board.ts";
import { Challenge } from "./challenge.ts";
import { Coherence } from "./coherence.ts";
import { Compare } from "./compare.ts";
import { Course } from "./course.ts";
import { Episode } from "./episode.ts";
import { Evaluation } from "./evaluation.ts";
import { Frontier } from "./frontier.ts";
import { Inspect } from "./inspect.ts";
import { Leaks } from "./leaks.ts";
import { Players } from "./players.ts";
import { Reach } from "./reach.ts";
import { Structures } from "./structures.ts";
import { Swap } from "./swap.ts";
import { Tensions } from "./tensions.ts";

/** Every Alpine component the course registers. */
export const COMPONENTS: Record<string, () => object> = {
  course: () => new Course(),
  inspect: () => new Inspect(),
  compare: () => new Compare(),
  reach: () => new Reach(),
  structures: () => new Structures(),
  board: () => new Board(),
  frontier: () => new Frontier(),
  players: () => new Players(),
  challenge: () => new Challenge(),
  episode: () => new Episode(),
  evaluation: () => new Evaluation(),
  swap: () => new Swap(),
  leaks: () => new Leaks(),
  tensions: () => new Tensions(),
  coherence: () => new Coherence(),
};
