;; This is just an exercise to translate my python code into lisp
;; code, and see how it goes.. It is just for fun.
(declaim (optimize (debug 3)))

(ql:quickload :alexandria)
(ql:quickload :fare-memoization)
(ql:quickload :py4cl)

(py4cl:import-module "numpy" :as "np")
(py4cl:import-module "warnings")

(sb-posix:chdir "/home/jin/pawel/")
;; (sb-posix:getcwd)

(defun range (min max &key (step 1))
  (loop for n from min below max by step
        collect n))

(defconstant +nan+ 'nan)
(defconstant +id-min+ 0)
(defconstant +id-max+ 673)
(defconstant +ids+ (range +id-min+ +id-max+))
(defconstant +frame-min+ 60)
(defconstant +frame-max+ 1320)
(defconstant +frames+ (range +frame-min+ +frame-max+ :step 10))

(defun file (frame)
  (concatenate 'string
               (sb-posix:getcwd) "/mnt/disp/disp_frame"
               (format nil "~a" frame) "_pointf_ref"))

(defun load-frame (frame)
  (let* ((input (py4cl:python-eval
                 (format nil "np.load(~s ,allow_pickle=True)" (file frame))))
         (result (make-hash-table :test #'equal))
         (idx (gethash "idx" input))
         (pointi (gethash "pointi" input))
         (pointf (gethash "pointf" input)))
    ;; Check data health.
    (assert (= (array-dimension idx 0)
               (array-dimension pointi 0)
               (array-dimension pointf 0)))
    ;; Set result.
    (dotimes (k (array-dimension idx 0))
      (let* ((x0 (aref pointi k 0))
             (y0 (aref pointi k 1))
             (x1 (aref pointf k 0))
             (y1 (aref pointf k 1))
             (dx (- x1 x0)) (dy (- y1 y0)))
        (setf (gethash (aref idx k) result)
              `(:x0 ,x0 :y0 ,y0 :dx ,dx :dy ,dy))))
    (loop for id in +ids+
          do (when (not (find id idx))
               (setf (gethash id result) +nan+)))
    result))
(fare-memoization:memoize 'load-frame)

(defun entry (frame id)
  (gethash id (load-frame frame)))
(fare-memoization:memoize 'entry)

(defun locations (id)
  (remove-duplicates (loop for frame in +frames+
                           unless (equal (entry frame id) +nan+)
                             collect (list (getf (entry frame id) :x0)
                                           (getf (entry frame id) :y0)))
                     :test #'equal))

(defun location (id)
  (let ((locs (locations id)))
    (case (length locs)
      (0 +nan+)
      (1 (car locs))
      (2 (warn "Multiple locations for id: ~a.~%" id)))))

(defun loc (id) (location id))

(defvar *null-ids*)
(defvar *non-null-ids*)
(defvar *x-mean*)
(defvar *y-mean*)

(defun compute ()
  (format t "Computing..~%")
  (setf *null-ids* (list)
        *non-null-ids* (list)
        *x-mean* 0
        *y-mean* 0)
  (loop for id in +ids+
        do (if (eql (loc id) +nan+)
               (push id *null-ids*)
               (push id *non-null-ids*)))
  (loop for id in *non-null-ids*
        sum (nth 0 (loc id)) into x-sum
        sum (nth 1 (loc id)) into y-sum
        finally (setf *x-mean* (/ x-sum (length *non-null-ids*))
                      *y-mean* (/ y-sum (length *non-null-ids*))))
  (format t "Computation ends!~%"))

;; FIXME Must be computed before the rest.. how to make it more robust?
(compute)

(defun euc-metric (floats)
  (loop for x being the elements of floats
        sum (expt x 2) into result
        finally (return (expt result 0.5))))

(defun mirror-candidates (id)
  (let* ((loc (loc id)) (x (nth 0 loc)) (y (nth 1 loc))
         (mirror-x (- (* 2 *x-mean*) x)) (mirror-y y)
         (result (copy-list *non-null-ids*)))
    (flet ((dist-to-mirror-loc (x y)
             (euc-metric (list (- x mirror-x) (- y mirror-y)))))
      (sort result
            (lambda (id1 id2)
              (<= (dist-to-mirror-loc (nth 0 (loc id1)) (nth 1 (loc id1)))
                  (dist-to-mirror-loc (nth 0 (loc id2)) (nth 1 (loc id2)))))))
    result))

(defun mirror (id)
  (nth 0 (mirror-candidates id)))
(fare-memoization:memoize 'mirror)

(defun stability (id)
  "Return the least n such that mirror^n(id) == mirror^(n+2)(id)."
  (format t "Computing stability for id: ~a.~%" id)
  (let ((n 0))
    (loop while (not (= id (mirror (mirror id))))
          do (sleep 0.05)
             (incf n)
             (setf id (mirror id))
          finally (return n))))

;; TODO the rest hasn't been done here.
