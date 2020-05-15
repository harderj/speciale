
import Data.List (transpose, elemIndex)
import Data.List.Extra (maximumOn)
import Data.Maybe (fromJust)
import Linear ((!*!), (!*), (*!), (*!!), (!!*), (!+!))

data Action = MoveUp | MoveDown | MoveLeft | MoveRight deriving (Show, Eq)
type Location = (Int, Int) -- x, y
type State = Location

-- deterministic transition kernel (function)
type TransF = State -> Action -> State

actualP :: TransF
actualP (2, 1) _ = (2, 5)
actualP (4, 1) _ = (4, 3)
actualP (x, y) MoveUp = (x, max(y-1) 1)
actualP (x, y) MoveDown = (x, min (y+1) 5)
actualP (x, y) MoveLeft = (max (x-1) 1, y)
actualP (x, y) MoveRight = (min (x+1) 5, y)

-- expected reward function
type ERewF = State -> Action -> Float

er :: ERewF
er (2, 1) _ = 10
er (4, 1) _ = 5
er (_, 1) MoveUp = -1
er (_, 5) MoveDown = -1
er (1, _) MoveLeft = -1
er (5, _) MoveRight = -1
er _ _ = 0

type Policy = State -> Action -> Float

tauU :: Policy
tauU _ _ = 0.25

-- value function
type Vfun = State -> Float

-- transition kernel
type TransK = State -> Action -> State -> Float

fromBool :: (Num a) => Bool -> a
fromBool b = if b then 1 else 0

transFToK :: TransF -> TransK
transFToK f = (\s a s' -> fromBool (s' == f s a))

pk :: TransK
pk = transFToK actualP

states = [ (x, y) | x <- [1..5], y <- [1..5] ]
actions = [ MoveUp, MoveDown, MoveLeft, MoveRight ]
sapairs = [ (s, a) | s <- states, a <- actions ]

gamma = 0.9

-- policy dependent T operator
tPol :: Policy -> Vfun -> Vfun
tPol tau v = (\s -> sum [ (f s a) * (tau s a) | a <- actions ])
   where f s a = er s a + gamma * (v (actualP s a))

v0 :: Vfun
v0 _ = 0

evalPol :: Int -> Policy -> Vfun
evalPol n tau = (iterate (tPol tauU) v0) !! n
-- too slow

evalPolStrict :: Int -> Vfun -> Vfun
evalPolStrict 0 x = x
evalPolStrict n x = evalPolStrict (n-1) $! (tPol tauU x)
-- not helping

showV :: Vfun -> String
showV v = unlines $ [ show [v (x, y) | x <- [1..5]] | y <- [1..5]]
printV = putStr . showV

-- trying with Matrices instead

a = [[0, 1],[0, 0]] :: [[Float]]
b = [[0, 0],[1, 0]] :: [[Float]]
d = [3, 4] :: [Float]
c = a !*! b
e = d *! a

vVec :: Vfun -> [[Float]]
vVec v = [[v s | s <- states]]

rVec :: ERewF -> [[Float]]
rVec r = [[r s a | s <- states, a <- actions]]

polMat :: Policy -> [[Float]]
polMat tau = [[ (fromBool $ s==s') * tau s a | s' <- states] | s <- states,
   a <- actions]

pMat :: TransK -> [[Float]]
pMat p = [[ p s a s' | (s, a) <- sapairs ] | s' <- states]

rM = rVec er
pM = pMat pk
tauUM = polMat tauU

s1 :: [Float]
s1 = [fromBool (s == (1,1)) | s <- states]
s2 :: [Float]
s2 = [fromBool (s == (3,4)) | s <- states]
s1a4 :: [Float]
s1a4 = [fromBool ((s, a) == ((1,1),MoveRight)) | s <- states, a <- actions]

tPolMat :: Policy -> [[Float]] -> [[Float]]
tPolMat tau v = (rM !+! gamma *!! v !*! pM) !*! (polMat tau)

resize5x5 (x1:x2:x3:x4:x5:xs) = [x1, x2, x3, x4, x5]:(resize5x5 xs)
resize5x5 _ = []

v0M = vVec v0
res1 = tPolMat tauU v0M
res100 = (iterate (tPolMat tauU) v0M) !! 100
v1M = [[1..25]] :: [[Float]]

dispVmat :: [[Float]] -> IO ()
dispVmat vM = mapM_ print $ transpose $ resize5x5 $ vM !! 0

toVfun :: [[Float]] -> Vfun
toVfun vM = \s -> vM !! 0 !! (fromJust $ elemIndex s states)

oneAction :: Action -> Policy
oneAction a = (\s a' -> fromBool (a == a'))

fromDetPol :: (State -> Action) -> Policy
fromDetPol f = \s a -> fromBool (a == f s)

greedyPol :: Vfun -> Policy
greedyPol v =
   fromDetPol (\s -> maximumOn (f s) actions)
   where f s a = (toVfun (tPolMat (oneAction a) (vVec v))) s

tBellMat :: [[Float]] -> [[Float]]
tBellMat v = tPolMat (greedyPol (toVfun v)) v

resBell2 = tBellMat $ tBellMat v0M

main = do dispVmat $ v0M
          dispVmat $ res1
          dispVmat $ res10
          dispVmat $ resBell2

-- tBellMat still too slow






